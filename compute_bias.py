"""Continuous F/T sensor bias (+ drift) estimation, implementing the full
pipeline of Nadeau et al., "Automated Continuous Force-Torque Sensor Bias
Estimation" (arXiv:2403.01068):

  1. Sec. 1.3 - a per-joint Kalman filter over [q, qdot, qddot] with a
     constant-acceleration / integrated-jerk-noise process model, run on the
     raw /franka/joint_states position+velocity measurements.
  2. Sec. 2 - forward kinematics mapping the filtered joint state to the
     sensor frame's world-frame linear/angular velocity and acceleration.
     Implemented with pinocchio (pin.getFrameVelocity /
     getFrameClassicalAcceleration, LOCAL_WORLD_ALIGNED) instead of hand-rolled
     screw axes / adjoint maps / Jacobian Hessian: both routes compute the
     same forward-kinematics quantities the paper defines in eqs. 28-29, and
     pinocchio's is the well-tested one already used elsewhere in this repo.
  3. Sec. 3 - a Kalman filter over state [bias(6), bias_drift(6)] that
     compares /ft_data against the wrench a known tool (mass, COM) should
     exert given the Sec. 2 kinematics (eq. 42's rigid-body regressor).

Known simplification: eq. 42's data-matrix uncertainty Sigma_D (propagated
from the joint-state KF's covariance through the kinematics) is treated as
negligible, so the bias KF's measurement noise is R_i = Sigma_W only. Doing
this exactly would require differentiating the full kinematic chain w.r.t.
[q, qdot, qddot], which the paper does not spell out either.

Sign convention note: this repo's bias (see ft_comparison.py) corrects a raw
reading with corrected = raw - bias, i.e. the negative of the paper's b (eq.
55 corrects with raw + b) -- handled in BiasKalmanFilter.update(). Separately,
matching this project's F/T sensor required negating eq. 42's wrench model
(expected_wrench() below) relative to the paper: eq. 42, as literally written,
predicts the support force/torque the robot exerts on the tool, while this
sensor reports the reaction the tool exerts on the sensor. This was verified
two ways: it reproduces the gravity-only formula already validated against
this sensor in ft_comparison.py, and it reproduces an independent
first-principles centripetal-force check for a rotating tool.
"""

import os
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda.urdf")
BIAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft_bias.txt")

GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])  # world = panda base frame
N_ARM_JOINTS = 7


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def rigid_body_data_matrix(a, omega, alpha):
    """6x10 regressor D such that D @ theta is the support wrench a rigid load
    requires, for a reference point (the sensor origin) with linear
    acceleration a, angular velocity omega and angular acceleration alpha (all
    expressed in the sensor frame), where
    theta = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] about the
    sensor origin. Eq. (42) of arXiv:2403.01068.
    """
    D = np.zeros((6, 10))

    D[0:3, 0] = a
    D[0:3, 1:4] = skew(omega) @ skew(omega) + skew(alpha)
    D[3:6, 1:4] = -skew(a)

    wx, wy, wz = omega
    ax, ay, az = alpha
    D[3, 4:10] = [ax, ay - wx * wz, az + wx * wy, -wy * wz, wy**2 - wz**2, wy * wz]
    D[4, 4:10] = [wx * wz, ax + wy * wz, wz**2 - wx**2, ay, az - wx * wy, -wx * wz]
    D[5, 4:10] = [-wx * wy, wx**2 - wy**2, ax - wy * wz, wx * wy, ay + wx * wz, az]
    return D


def inertial_params(mass, com, inertia=None):
    """theta = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] about the sensor origin."""
    if inertia is None:
        inertia = np.zeros(6)
    return np.concatenate([[mass], mass * np.asarray(com, dtype=float), inertia])


def expected_wrench(R, a_world, omega_world, alpha_world, theta, gravity_world=GRAVITY_WORLD):
    """Wrench the tool (inertial params theta) is expected to make the sensor
    read, given the sensor frame's world-frame kinematics. See the sign
    convention note in the module docstring for the leading minus sign.
    """
    a_proper = R.T @ (a_world - gravity_world)
    omega = R.T @ omega_world
    alpha = R.T @ alpha_world
    D = rigid_body_data_matrix(a_proper, omega, alpha)
    return -(D @ theta)


class JointKalmanFilter:
    """Per-joint [q, qdot, qddot] KF: constant-acceleration motion model with
    integrated-jerk process noise (Sec. 1.3, eqs. 10-13), position+velocity
    observation model (eq. 2 -- the only one the paper gives in full; Sec. 1.3
    reuses the same structure).
    """

    def __init__(self, q0, sigma_jerk=10.0, eta_q=1e-4, eta_qdot=1e-2):
        self.sigma_jerk2 = sigma_jerk**2
        self.x = np.array([q0, 0.0, 0.0])
        self.P = np.diag([1e-4, 1.0, 10.0])
        self.C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.R = np.diag([eta_q**2, eta_qdot**2])

    def predict(self, dt):
        A = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        Q = self.sigma_jerk2 * np.array([
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ])
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def update(self, q_meas, qdot_meas):
        y = np.array([q_meas, qdot_meas])
        innovation = y - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(3) - K @ self.C) @ self.P


class JointStatesEstimator:
    """Runs one JointKalmanFilter per arm joint (Sec. 1) and returns filtered
    [q, qdot, qddot] arrays for all N_ARM_JOINTS joints.
    """

    def __init__(self, n_joints=N_ARM_JOINTS, **kf_kwargs):
        self.n_joints = n_joints
        self.kf_kwargs = kf_kwargs
        self.filters = None
        self.last_time = None

    def update(self, position, velocity, stamp):
        position = np.asarray(position[:self.n_joints], dtype=float)
        velocity = np.asarray(velocity[:self.n_joints], dtype=float)

        if self.filters is None:
            self.filters = [JointKalmanFilter(position[j], **self.kf_kwargs) for j in range(self.n_joints)]
            self.last_time = stamp
            return position.copy(), velocity.copy(), np.zeros(self.n_joints)

        dt = max(stamp - self.last_time, 1e-4)
        self.last_time = stamp

        q = np.zeros(self.n_joints)
        qdot = np.zeros(self.n_joints)
        qddot = np.zeros(self.n_joints)
        for j, kf in enumerate(self.filters):
            kf.predict(dt)
            kf.update(position[j], velocity[j])
            q[j], qdot[j], qddot[j] = kf.x
        return q, qdot, qddot


def sensor_kinematics(model, data, frame_id, q7, qdot7, qddot7):
    """Forward kinematics of the sensor frame, in world-frame coordinates
    (Sec. 2, eqs. 27-29), via pinocchio instead of the paper's hand-rolled
    screw-theory Jacobian/Hessian -- see the module docstring.
    """
    q = np.zeros(model.nq)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    q[:N_ARM_JOINTS] = q7
    v[:N_ARM_JOINTS] = qdot7
    a[:N_ARM_JOINTS] = qddot7

    pin.forwardKinematics(model, data, q, v, a)
    pin.updateFramePlacements(model, data)

    R = data.oMf[frame_id].rotation
    vel = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    acc = pin.getFrameClassicalAcceleration(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return R, vel.angular.copy(), acc.linear.copy(), acc.angular.copy()


class BiasKalmanFilter:
    """Kalman filter over x = [bias(6), bias_drift(6)] (eqs. 31-54 of
    arXiv:2403.01068).

    Sign convention: this repo corrects a raw reading with
    corrected = raw - bias (see ft_comparison.py), the negative of the paper's
    b (which corrects with raw + b, eq. 55). The filter is linear, so that's
    just an observation-sign flip, handled in update().
    """

    def __init__(self, sigma_q=1e-3, bias0=None):
        self.sigma_q = sigma_q
        self.x = np.zeros(12) if bias0 is None else np.concatenate([bias0, np.zeros(6)])
        self.P = np.diag([1.0] * 6 + [1e-2] * 6)
        self.C = np.hstack([np.eye(6), np.zeros((6, 6))])

    def predict(self, dt):
        Qc = self.sigma_q**2 * np.eye(6)
        A = np.block([[np.eye(6), dt * np.eye(6)], [np.zeros((6, 6)), np.eye(6)]])
        Q = np.block([[dt**3 / 3 * Qc, dt**2 / 2 * Qc], [dt**2 / 2 * Qc, dt * Qc]])
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def update(self, measured, expected, R):
        y = measured - expected  # observes bias (repo sign) + noise
        innovation = y - self.C @ self.x
        S = self.C @ self.P @ self.C.T + R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(12) - K @ self.C) @ self.P

    @property
    def bias(self):
        return self.x[:6]

    @property
    def drift(self):
        return self.x[6:]


class BiasEstimatorNode(Node):
    def __init__(self):
        super().__init__('ft_bias_estimator')

        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("ft_sensor")

        # Known inertial parameters of whatever is currently mounted on the
        # sensor -- update these to match the tool used for calibration.
        self.object_mass = 0.075  # kg
        self.object_COM = np.array([0.0, 0.0, 0.05])  # m, in ft_sensor frame
        self.theta = inertial_params(self.object_mass, self.object_COM)

        # Tunable noise parameters -- defaults are placeholders, tune to your
        # robot/sensor.
        self.joint_estimator = JointStatesEstimator(sigma_jerk=10.0, eta_q=1e-4, eta_qdot=1e-2)
        self.R_wrench = np.diag([0.05**2] * 3 + [0.01**2] * 3)  # Sigma_W
        self.kf = BiasKalmanFilter(sigma_q=1e-3, bias0=self.load_initial_bias())

        self.expected = None  # latest expected wrench from joint_states + known theta
        self.latest_measured = None  # latest raw /ft_data reading
        self.last_kf_time = None
        self.last_ft_time = None
        self.last_joint_time = None
        self.data_timeout = 0.5  # s without a message before a topic counts as not publishing
        self.data_status = ''
        self.last_save_time = 0.0
        self.save_period = 1.0  # s

        # Plot history: each entry is (t, measured[6], expected[6], bias[6], drift[6]),
        # sampled on a fixed timer so the plot keeps scrolling (with NaN gaps) even if
        # a topic stalls, same pattern as ft_comparison.py.
        self.sample_period = 0.1  # s
        self.plot_window = 30.0  # s of history shown
        self.buffer = deque(maxlen=int(self.plot_window / self.sample_period))
        self.start_time = time.time()

        self.create_subscription(JointState, '/franka/joint_states', self.joint_callback, 10)
        self.create_subscription(WrenchStamped, '/ft_data', self.wrench_callback, 10)
        self.sample_timer = self.create_timer(self.sample_period, self.sample_callback)
        self.get_logger().info('Initialized FT bias estimator (arXiv:2403.01068, full pipeline)')

    def load_initial_bias(self):
        if not os.path.exists(BIAS_PATH):
            return None
        try:
            bias = np.loadtxt(BIAS_PATH)
            if bias.shape == (6,):
                return bias
        except (OSError, ValueError):
            pass
        return None

    def joint_callback(self, msg: JointState):
        if len(msg.position) < N_ARM_JOINTS or len(msg.velocity) < N_ARM_JOINTS:
            self.get_logger().warn(
                f'Expected at least {N_ARM_JOINTS} joint positions and velocities, '
                f'got {len(msg.position)} / {len(msg.velocity)}')
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp == 0.0:
            stamp = time.time()

        q, qdot, qddot = self.joint_estimator.update(msg.position, msg.velocity, stamp)
        R, omega_world, a_world, alpha_world = sensor_kinematics(
            self.model, self.data, self.ee_frame_id, q, qdot, qddot)
        self.expected = expected_wrench(R, a_world, omega_world, alpha_world, self.theta)
        self.last_joint_time = time.time()

    def wrench_callback(self, msg: WrenchStamped):
        w = msg.wrench
        measured = np.array([w.force.x, w.force.y, w.force.z,
                              w.torque.x, w.torque.y, w.torque.z])
        self.latest_measured = measured
        self.last_ft_time = time.time()

        if self.expected is None:
            return  # wait for the first joint_states message

        now = time.time()
        dt = 0.0 if self.last_kf_time is None else now - self.last_kf_time
        self.last_kf_time = now
        if dt > 0:
            self.kf.predict(dt)
        self.kf.update(measured, self.expected, self.R_wrench)

        if now - self.last_save_time > self.save_period:
            self.save_bias()
            self.last_save_time = now

    def latest_or_nan(self, value, last_time, topic, now):
        """Return value if the topic is fresh, otherwise NaNs (plotted as a gap)."""
        if last_time is None:
            self.get_logger().warn(f'Waiting for data on {topic}...', throttle_duration_sec=2.0)
            return np.full(6, np.nan), f'{topic}: waiting'
        age = now - last_time
        if age > self.data_timeout:
            self.get_logger().error(
                f'{topic} not publishing (last message {age:.1f} s ago)',
                throttle_duration_sec=2.0)
            return np.full(6, np.nan), f'{topic}: no data for {age:.0f} s'
        return value, None

    def sample_callback(self):
        now = time.time()
        measured, ft_problem = self.latest_or_nan(
            self.latest_measured, self.last_ft_time, '/ft_data', now)
        expected, joint_problem = self.latest_or_nan(
            self.expected, self.last_joint_time, '/franka/joint_states', now)
        self.data_status = '   '.join(p for p in (ft_problem, joint_problem) if p)
        self.buffer.append((now - self.start_time, measured, expected,
                             self.kf.bias.copy(), self.kf.drift.copy()))

    def save_bias(self):
        try:
            np.savetxt(BIAS_PATH, self.kf.bias[None],
                       header=f'FT bias [Fx Fy Fz Tx Ty Tz], saved {time.strftime("%Y-%m-%d %H:%M:%S")} '
                              f'(drift {np.round(self.kf.drift, 5).tolist()})')
        except OSError as e:
            self.get_logger().error(f'Failed to save bias to {BIAS_PATH}: {e}')
        self.get_logger().info(
            f'bias={np.round(self.kf.bias, 4).tolist()} drift={np.round(self.kf.drift, 5).tolist()}')


def run_plotter(node):
    plt.ion()
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle('Raw Measured vs Expected Wrench, and Estimated Bias (arXiv:2403.01068)', fontsize=13)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], hspace=0.4, wspace=0.3)

    titles = ['Force X (N)', 'Force Y (N)', 'Force Z (N)',
              'Torque X (Nm)', 'Torque Y (Nm)', 'Torque Z (Nm)']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    measured_lines = []
    expected_lines = []
    wrench_axes = []
    for i, (title, c) in enumerate(zip(titles, colors)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        measured_lines.append(ax.plot([], [], f'{c}-', label='raw measured')[0])
        expected_lines.append(ax.plot([], [], 'k--', label='expected')[0])
        ax.set_title(title)
        ax.grid(True)
        wrench_axes.append(ax)
    wrench_axes[0].legend(loc='upper left', fontsize=8)

    bias_ax = fig.add_subplot(gs[2, :])
    bias_ax.set_title('Estimated bias [Fx Fy Fz Tx Ty Tz]')
    bias_ax.grid(True)
    bias_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    bias_lines = [bias_ax.plot([], [], c, label=label)[0]
                  for label, c in zip(bias_labels, colors)]
    bias_ax.legend(loc='upper left', ncol=6, fontsize=8)

    data_text = fig.text(0.99, 0.01, '', va='bottom', ha='right', color='r')

    while rclpy.ok():
        entries = list(node.buffer)
        if entries:
            t_data = [e[0] for e in entries]
            measured = np.array([e[1] for e in entries])
            expected = np.array([e[2] for e in entries])
            bias = np.array([e[3] for e in entries])

            for i, (m_line, e_line) in enumerate(zip(measured_lines, expected_lines)):
                m_line.set_data(t_data, measured[:, i])
                e_line.set_data(t_data, expected[:, i])
                m_line.axes.relim()
                m_line.axes.autoscale_view()

            for i, b_line in enumerate(bias_lines):
                b_line.set_data(t_data, bias[:, i])
            bias_ax.relim()
            bias_ax.autoscale_view()

            xlim = (max(0.0, t_data[-1] - node.plot_window), max(node.plot_window, t_data[-1]))
            for ax in wrench_axes + [bias_ax]:
                ax.set_xlim(*xlim)

        data_text.set_text(node.data_status)
        plt.pause(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = BiasEstimatorNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_plotter(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_bias()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
