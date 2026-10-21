import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt

model = pin.buildModelFromUrdf("panda.urdf")

data = model.createData()

q = np.zeros(9)
q[:7] = [0.012764, -0.772067, 0.005929, -2.398398, -0.031545, 1.660237, 0.838603]

pin.forwardKinematics(model, data, q)

ee_frame_id = model.getFrameId("ft_sensor")

ee_pose = pin.updateFramePlacement(model ,data, ee_frame_id)

ee_position = ee_pose.translation
ee_rotation = ee_pose.rotation

print("EE Position relative to base: ", ee_position)
print("EE Rotation relative to base: ", ee_rotation)

world_grav = [0, 0, -9.81] #world=panda base frame

transposed_R = [list(row) for row in zip(*ee_rotation)]
transposed_R = np.matrix(transposed_R)
grav_at_ee = ee_rotation.T @ np.array(world_grav)

print("grav at ee: ", grav_at_ee)

object_mass = 0.0 #kg
object_COM = [0, 0, 0.1] #m

forces_expected_ee = object_mass * grav_at_ee
torques_expected_ee = np.cross(object_COM, forces_expected_ee)

print("forces: ", forces_expected_ee)
print("torques: ", torques_expected_ee)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
points = []
for j in range(0, 8):          # base + 7 arm joints, no fingers
    M = data.oMi[j]
    points.append(M.translation)
    for k, c in enumerate("rgb"):
        ax.quiver(*M.translation, *M.rotation[:, k], color=c, length=0.1)
hand_id = model.getFrameId("ft_sensor")
hand_pose = pin.updateFramePlacement(model, data, hand_id)
points.append(hand_pose.translation)
for k, c in enumerate("rgb"):
    ax.quiver(*hand_pose.translation, *hand_pose.rotation[:, k], color=c, length=0.1)

ax.plot(*zip(*points), "k-o")  # stick figure through all joint origins
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 1)
ax.set_box_aspect([1, 1, 1])
plt.show()