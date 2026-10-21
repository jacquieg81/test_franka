import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time

class WrenchPlotterNode(Node):
    def __init__(self):
        super().__init__('wrench_plotter_node')
        
        # Buffer up to 100 data points (~10-50 seconds depending on pub rate)
        self.max_len = 100
        self.time_buffer = deque(maxlen=self.max_len)
        self.fx = deque(maxlen=self.max_len)
        self.fy = deque(maxlen=self.max_len)
        self.fz = deque(maxlen=self.max_len)
        self.tx = deque(maxlen=self.max_len)
        self.ty = deque(maxlen=self.max_len)
        self.tz = deque(maxlen=self.max_len)
        
        self.start_time = time.time()
        
        # Subscribe to /sensor/wrench
        self.subscription = self.create_subscription(
            WrenchStamped,
            '/ft_data',
            self.wrench_callback,
            10
        )

    def wrench_callback(self, msg: WrenchStamped):
        t = time.time() - self.start_time
        self.time_buffer.append(t)
        self.fx.append(msg.wrench.force.x)
        self.fy.append(msg.wrench.force.y)
        self.fz.append(msg.wrench.force.z)
        self.tx.append(msg.wrench.torque.x)
        self.ty.append(msg.wrench.torque.y)
        self.tz.append(msg.wrench.torque.z)


def run_plotter(node):
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('6-Axis Force / Torque Sensor Data (/ft_data)', fontsize=14)

    lines = [
        axs[0, 0].plot([], [], 'r-')[0],  # Fx
        axs[0, 1].plot([], [], 'g-')[0],  # Fy
        axs[0, 2].plot([], [], 'b-')[0],  # Fz
        axs[1, 0].plot([], [], 'c-')[0],  # Tx
        axs[1, 1].plot([], [], 'm-')[0],  # Ty
        axs[1, 2].plot([], [], 'y-')[0]   # Tz
    ]
    
    titles = ['Force X (N)', 'Force Y (N)', 'Force Z (N)', 
              'Torque X (Nm)', 'Torque Y (Nm)', 'Torque Z (Nm)']

    for idx, ax in enumerate(axs.flat):
        ax.set_title(titles[idx])
        ax.grid(True)

    while rclpy.ok():
        t_data = list(node.time_buffer)
        if len(t_data) > 0:
            data_lists = [list(node.fx), list(node.fy), list(node.fz),
                          list(node.tx), list(node.ty), list(node.tz)]
            
            for line, data in zip(lines, data_lists):
                line.set_data(t_data, data)
                line.axes.relim()
                line.axes.autoscale_view()
            
            plt.pause(0.05)
        else:
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = WrenchPlotterNode()

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
