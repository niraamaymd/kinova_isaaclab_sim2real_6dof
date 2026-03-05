import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration

class Gen3OnnxPolicy(Node):
    def __init__(self):
        super().__init__('gen3_onnx_policy')

        # 1. Load ONNX Model
        model_path = 'kinova_gen3_policy.onnx'
        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.get_logger().info(f"Successfully loaded ONNX model: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            return

        # 2. State Variables
        self.current_joint_pos = np.zeros(6, dtype=np.float32)
        self.current_joint_vel = np.zeros(6, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

        # IsaacLab Configured Target Range: X(0.35-0.65), Y(-0.2, 0.2), Z(0.15-0.5)
        self.pose_command = np.array([0.45, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 3. Training Order (Crucial for IsaacLab consistency)
        self.TRAINING_ORDER = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

        # 4. Subscribers & Publishers
        self.create_subscription(JointTrajectoryControllerState, '/joint_trajectory_controller/state', self.state_callback, 10)
        self.create_subscription(PoseStamped, '/target_pose', self.target_callback, 10)
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        # 5. Timer (100Hz)
        self.timer = self.create_timer(0.01, self.step_callback)
        self.get_logger().info("Policy Node Started. Waiting for state updates...")

    def state_callback(self, msg: JointTrajectoryControllerState):
        """Map incoming ROS joints to the order expected by the ONNX model."""
        for i, name in enumerate(msg.joint_names):
            if name in self.TRAINING_ORDER:
                idx = self.TRAINING_ORDER.index(name)
                self.current_joint_pos[idx] = msg.actual.positions[i]
                self.current_joint_vel[idx] = msg.actual.velocities[i]

    def target_callback(self, msg: PoseStamped):
        p = msg.pose
        self.pose_command = np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z
        ], dtype=np.float32)
        self.get_logger().info(f"Target Updated: {self.pose_command[:3]}")

    def step_callback(self):
        # 1. Construct Observation
        # Ensure this matches the EXACT order used in training
        obs = np.concatenate([
            self.current_joint_pos, # 6
            self.current_joint_vel, # 6
            self.pose_command,      # 7
            self.last_action        # 6
        ]).astype(np.float32).reshape(1, 25)

        # 2. Inference
        input_name = self.ort_session.get_inputs()[0].name
        raw_action = self.ort_session.run(None, {input_name: obs})[0].flatten()

        # --- THE FIX FOR STEP 4 ---
        # In RL, 'last_action' in the NEXT step must be the 'current_action' of THIS step.
        # Store the raw, unscaled action from the model output.
        self.last_action = raw_action

        # 3. Scale for the REAL hardware
        # If it's a delta policy, we add the nudge to the current real position
        scale = 0.001  # Start very small!
        target_positions = self.current_joint_pos + (raw_action * scale)

        # 5. Publish
        traj = JointTrajectory()
        traj.joint_names = self.TRAINING_ORDER # Ensure command matches training order
        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=20000000) # 20ms
        
        traj.points.append(point)
        self.pub.publish(traj)

# --- THE MISSING PIECE ---
def main(args=None):
    rclpy.init(args=args)
    node = Gen3OnnxPolicy()
    try:
        rclpy.spin(node) # This keeps the script running!
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

