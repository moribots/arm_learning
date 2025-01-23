import pybullet as p
import pybullet_data
import time
import numpy as np

def main():
    # Initialize PyBullet simulation
    physicsClient = p.connect(p.GUI)  # Start in GUI mode
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set PyBullet data path

    # Set up the environment
    p.setGravity(0, 0, -9.8)  # Add gravity
    planeId = p.loadURDF("plane.urdf")  # Load the ground plane

    # Load a 7-DOF robotic arm (Franka Emika Panda)
    robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Print joint information to understand the structure
    num_joints = p.getNumJoints(robotId)
    print("Number of joints:", num_joints)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')} (Type: {joint_info[2]})")

    # Create shelving environment with a ceiling
    shelf_height = 0.15
    shelf_spacing = 0.25
    num_shelves = 4
    shelf_width = 0.8
    shelf_depth = 0.3

    # Create shelves and ceiling
    for i in range(num_shelves):
        shelf_z = shelf_height / 2 + i * (shelf_height + shelf_spacing)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0.0, 0.5, shelf_z])

    # Define target joint positions for motion
    target_positions = [0.5, -0.5, 0.5, -1.0, 0.5, 1.0, 0.2]  # Example target positions

    # Move joints using Position Control
    for joint_index in range(num_joints):
        if p.getJointInfo(robotId, joint_index)[2] == p.JOINT_REVOLUTE:  # Skip fixed joints
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL, target_positions[joint_index])

    # Run the simulation for a fixed duration
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # Simulation step at 240 Hz

    # Move the arm to a goal position smoothly
    move_to_goal(robotId, num_joints, [0.2, -0.2, 0.4, -0.6, 0.2, 0.8, 0.1])

    # Clean up
    p.disconnect()

def move_to_goal(robotId, num_joints, goal_positions, steps=100):
    """Smoothly move the robot arm to the goal positions."""
    current_positions = [p.getJointState(robotId, i)[0] for i in range(num_joints)]
    for step in range(steps):
        interpolated_positions = current_positions + (np.array(goal_positions) - np.array(current_positions)) * (step / steps)
        for joint_index in range(num_joints):
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL, interpolated_positions[joint_index])
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

if __name__ == "__main__":
    main()
