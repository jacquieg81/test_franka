import os
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda.urdf")
BIAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft_bias.txt")

class FTComparisonNode(Node):

    def __init__(self):
        super().__init__('ft_comparison_node')

        # Build the kinematic model once (nq = 9: 7 arm joints + 2 fingers)
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("ft_sensor")

        self.world_grav = np.array([0, 0, -9.81])  # world = panda base frame
        self.object_mass = 0.075  # kg
        self.object_COM = np.array([0, 0, 0.05])  # m, in ft_sensor frame

        # Latest wrenches [Fx, Fy, Fz, Tx, Ty, Tz], None until first message
        self.expected_wrench = None
        self.measured_wrench = None
        self.last_ft_time = None
        self.last_joint_time = None
        self.data_timeout = 0.5  # s without a message before a topic counts as not publishing
        self.data_status = ''

        # FT bias, subtracted from every raw reading. Triggered from the plot window;
        # take it with no object mounted so it only removes the sensor offset.
        # The most recent bias is saved to BIAS_PATH and reloaded on startup.
        self.bias = np.zeros(6)
        self.bias_num_samples = 100
        self.bias_lock = threading.Lock()
        self.bias_samples = None  # list while collecting, None otherwise
        self.bias_status = 'not biased'
        self.load_bias()

        # Each entry is (t, measured[6], expected[6]) so the plot thread always
        # sees matching measured/expected pairs. Measured/expected are NaN while
        # /ft_data or /franka/joint_states is down, and resume when data comes back.
        self.sample_period = 0.1  # s
        self.plot_window = 30.0  # s of history shown
        self.buffer = deque(maxlen=int(self.plot_window / self.sample_period))
        self.start_time = time.time()

        self.joint_sub = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_callback,
            10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/ft_data',
            self.wrench_callback,
            10)
        self.sample_timer = self.create_timer(self.sample_period, self.sample_callback)
        self.get_logger().info('Initialized FT comparison node!')

    def joint_callback(self, msg):
        if len(msg.position) < 7:
            self.get_logger().warn(f'Expected at least 7 joint positions, got {len(msg.position)}')
            return

        ee_pose = self.compute_ee_pose(msg.position[:7])
        forces, torques = self.compute_expected_wrench(ee_pose)
        self.expected_wrench = np.concatenate([forces, torques])
        self.last_joint_time = time.time()

    def wrench_callback(self, msg: WrenchStamped):
        w = msg.wrench
        if w is None:
            self.get_logger().warn('Received WrenchStamped with no wrench data')
            return
        raw = np.array([w.force.x, w.force.y, w.force.z,
                        w.torque.x, w.torque.y, w.torque.z])
        self.last_ft_time = time.time()

        with self.bias_lock:
            if self.bias_samples is not None:
                self.bias_samples.append(raw)
                if len(self.bias_samples) >= self.bias_num_samples:
                    self.finish_bias()

        self.measured_wrench = raw - self.bias

    def request_bias(self):
        with self.bias_lock:
            if self.bias_samples is not None:
                self.get_logger().warn('Bias already in progress')
                return
            self.bias_samples = []
            self.bias_status = f'biasing ({self.bias_num_samples} samples)...'
        self.get_logger().info(f'Collecting {self.bias_num_samples} FT samples for bias, keep the robot still')

    def finish_bias(self):
        # Called with bias_lock held
        samples = np.array(self.bias_samples)
        self.bias = samples.mean(axis=0)
        self.bias_samples = None
        std = samples.std(axis=0)
        self.bias_status = 'biased: ' + ' '.join(f'{b:+.2f}' for b in self.bias)
        self.get_logger().info(
            f'Bias set [Fx Fy Fz Tx Ty Tz] = {np.round(self.bias, 4).tolist()} '
            f'(std {np.round(std, 4).tolist()})')
        self.save_bias()

    def save_bias(self):
        try:
            np.savetxt(BIAS_PATH, self.bias[None],
                       header=f'FT bias [Fx Fy Fz Tx Ty Tz], saved {time.strftime("%Y-%m-%d %H:%M:%S")}')
            self.get_logger().info(f'Saved bias to {BIAS_PATH}')
        except OSError as e:
            self.get_logger().error(f'Failed to save bias to {BIAS_PATH}: {e}')

    def load_bias(self):
        if not os.path.exists(BIAS_PATH):
            self.get_logger().warn(f'No saved bias at {BIAS_PATH}, press b to bias')
            return
        try:
            bias = np.loadtxt(BIAS_PATH)
            if bias.shape != (6,):
                raise ValueError(f'expected 6 values, got shape {bias.shape}')
        except (OSError, ValueError) as e:
            self.get_logger().error(f'Failed to load bias from {BIAS_PATH}: {e}')
            return
        self.bias = bias
        self.bias_status = 'loaded bias: ' + ' '.join(f'{b:+.2f}' for b in self.bias)
        self.get_logger().info(f'Loaded bias {np.round(self.bias, 4).tolist()} from {BIAS_PATH}')

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
            self.measured_wrench, self.last_ft_time, '/ft_data', now)
        expected, joint_problem = self.latest_or_nan(
            self.expected_wrench, self.last_joint_time, '/franka/joint_states', now)
        self.data_status = '   '.join(p for p in (ft_problem, joint_problem) if p)

        # A bias needs uninterrupted FT data, so abort it if the sensor drops out
        if ft_problem and self.last_ft_time is not None:
            with self.bias_lock:
                if self.bias_samples is not None:
                    self.bias_samples = None
                    self.bias_status = 'bias aborted: /ft_data stopped, press b to retry'
                    self.get_logger().error('Bias aborted: /ft_data stopped publishing')

        self.buffer.append((now - self.start_time, measured, expected))

    def compute_ee_pose(self, arm_positions):
        q = np.zeros(self.model.nq)
        q[:7] = arm_positions

        pin.forwardKinematics(self.model, self.data, q)
        return pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)

    def compute_expected_wrench(self, ee_pose):
        grav_at_ee = ee_pose.rotation.T @ self.world_grav

        forces = self.object_mass * grav_at_ee
        torques = np.cross(self.object_COM, forces)
        return forces, torques


def run_plotter(node):
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Measured (/ft_data) vs Expected Wrench', fontsize=14)

    titles = ['Force X (N)', 'Force Y (N)', 'Force Z (N)',
              'Torque X (Nm)', 'Torque Y (Nm)', 'Torque Z (Nm)']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    measured_lines = []
    expected_lines = []
    for ax, title, c in zip(axs.flat, titles, colors):
        measured_lines.append(ax.plot([], [], f'{c}-', label='measured')[0])
        expected_lines.append(ax.plot([], [], 'k--', label='expected')[0])
        ax.set_title(title)
        ax.grid(True)
    axs[0, 0].legend(loc='upper left')

    # Bias controls: click the button or press 'b' with the plot window focused
    fig.subplots_adjust(bottom=0.15)
    bias_ax = fig.add_axes([0.02, 0.02, 0.15, 0.06])
    bias_button = Button(bias_ax, "Bias FT ('b')")
    bias_button.on_clicked(lambda _event: node.request_bias())
    fig.canvas.mpl_connect('key_press_event',
                           lambda event: node.request_bias() if event.key == 'b' else None)
    status_text = fig.text(0.19, 0.045, '', va='center', family='monospace')
    data_text = fig.text(0.98, 0.045, '', va='center', ha='right', color='r')

    while rclpy.ok():
        entries = list(node.buffer)
        if entries:
            t_data = [e[0] for e in entries]
            measured = np.array([e[1] for e in entries])
            expected = np.array([e[2] for e in entries])

            for i, (m_line, e_line) in enumerate(zip(measured_lines, expected_lines)):
                m_line.set_data(t_data, measured[:, i])
                e_line.set_data(t_data, expected[:, i])
                m_line.axes.relim()
                m_line.axes.autoscale_view()
                # Keep the x-axis scrolling even while a topic is down (all-NaN data)
                m_line.axes.set_xlim(max(0.0, t_data[-1] - node.plot_window),
                                     max(node.plot_window, t_data[-1]))

        status_text.set_text(node.bias_status)
        data_text.set_text(node.data_status)
        plt.pause(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = FTComparisonNode()

    # Spin ROS 2 in a background thread so Matplotlib UI can run on the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_plotter(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
