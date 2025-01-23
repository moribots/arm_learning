import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RobotEnvironment:
	def __init__(self):
		# Connect to PyBullet simulation in GUI mode
		# self.physicsClient = p.connect(p.GUI)
		self.physicsClient = p.connect(p.DIRECT)
		# Set the search path for PyBullet URDF models
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		# Set gravity for the simulation environment
		p.setGravity(0, 0, -9.8)
		# Load a flat plane as the ground
		self.planeId = p.loadURDF("plane.urdf")
		# Load the Franka Panda robot arm in a fixed base configuration
		self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
		# Retrieve the number of joints in the robot
		self.num_joints = p.getNumJoints(self.robotId)
		# Set up the shelving environment
		self.setup_shelves()

	def setup_shelves(self):
		# Define the dimensions and spacing of the shelves
		shelf_height = 0.2
		shelf_spacing = 0.15
		num_shelves = 4
		shelf_width = 0.8
		shelf_depth = 0.3
		ceiling_height = shelf_height * num_shelves + shelf_spacing * (num_shelves - 1) + 0.1

		# Create shelves and ceiling
		for i in range(num_shelves):
			shelf_z = shelf_height / 2 + i * (shelf_height + shelf_spacing)
			collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2])
			visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1])
			p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[0.0, 0.5, shelf_z])

	def generate_voxel_grid(self, grid_resolution=0.05):
		# Generate a 3D voxel grid representation of the environment
		voxel_grid = []
		x_range = np.arange(-0.5, 0.5, grid_resolution)
		y_range = np.arange(-0.5, 1.0, grid_resolution)
		z_range = np.arange(0, 1.5, grid_resolution)

		for x in x_range:
			for y in y_range:
				for z in z_range:
					# Check if a point is inside an obstacle (shelves)
					closest_points = p.getClosestPoints(self.planeId, self.robotId, distance=grid_resolution, linkIndexA=-1, linkIndexB=-1)
					is_occupied = any([point[8] < grid_resolution for point in closest_points])
					voxel_grid.append((x, y, z, is_occupied))
		return torch.tensor(voxel_grid, dtype=torch.float32)

	def reset(self):
		# Reset the robot's base position and orientation to its default state
		p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0], [0, 0, 0, 1])
		# Reset each joint to its default position
		for joint_index in range(self.num_joints):
			p.resetJointState(self.robotId, joint_index, 0)

	def get_state(self):
		# Retrieve the current positions and velocities of all joints
		state = []
		for joint_index in range(self.num_joints):
			joint_state = p.getJointState(self.robotId, joint_index)
			state.append(joint_state[0])  # Position
			state.append(joint_state[1])  # Velocity

		# Get voxel grid representation of obstacles
		voxel_grid = self.generate_voxel_grid().flatten()

		# Combine robot state with voxel grid
		combined_state = torch.cat((torch.tensor(state, dtype=torch.float32), voxel_grid.float()))
		return combined_state

	def apply_action(self, action):
		# Apply the given action (torques) to all joints
		for joint_index in range(self.num_joints):
			p.setJointMotorControl2(self.robotId, joint_index, p.TORQUE_CONTROL, force=action[joint_index])
		# Step the simulation forward
		p.stepSimulation()
		# Sleep to maintain a simulation step frequency of 240 Hz
		# time.sleep(1.0 / 240.0)

	def disconnect(self):
		# Disconnect from the PyBullet simulation
		p.disconnect()

class ActorCriticNetwork(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(ActorCriticNetwork, self).__init__()
		# Define a fully connected neural network with two hidden layers
		self.fc1 = nn.Linear(state_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		# Actor head outputs actions
		self.actor = nn.Linear(128, action_dim)
		# Critic head outputs the state value
		self.critic = nn.Linear(128, 1)

	def forward(self, state):
		# Forward pass through the network
		x = torch.relu(self.fc1(state))
		x = torch.relu(self.fc2(x))
		action_logits = torch.tanh(self.actor(x))  # Actor outputs action logits
		state_value = self.critic(x)  # Critic outputs state value
		return action_logits, state_value

class PPOAgent:
	def __init__(self, state_dim, action_dim):
		# Initialize the PPO agent with a policy network
		self.policy = ActorCriticNetwork(state_dim, action_dim)
		# Use Adam optimizer to train the network
		self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)
		self.gamma = 0.99
		self.eps_clip = 0.2
		self.k_epochs = 4

	def select_action(self, state):
		# Forward pass through the policy network to get actions
		action_logits, _ = self.policy(state)
		return action_logits.detach().numpy()

	def train(self, states, actions, rewards, values):
		# Compute advantages
		advantages = []
		discounted_sum = 0
		for reward, value in zip(reversed(rewards), reversed(values)):
			td_error = reward + self.gamma * discounted_sum - value
			discounted_sum = value + td_error
			advantages.insert(0, td_error)

		# Convert advantages to float32
		advantages = [torch.tensor(a, dtype=torch.float32) for a in advantages]

		# Update the policy and value networks
		for _ in range(self.k_epochs):
			for state, action, advantage in zip(states, actions, advantages):
				state = state.float()  # Ensure state is float32
				action = torch.tensor(action, dtype=torch.float32)  # Ensure action is float32
				advantage = advantage.float()  # Ensure advantage is float32

				action_logits, state_value = self.policy(state)
				action_prob = torch.sum(action_logits * action)
				ratio = action_prob / (torch.sum(action) + 1e-8)

				# Clipped Surrogate Objective
				surr1 = ratio * advantage
				surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
				policy_loss = -torch.min(surr1, surr2).mean()

				# Value Loss
				value_loss = nn.MSELoss()(state_value, torch.tensor([advantage.item() + state_value.item()], dtype=torch.float32))

				loss = policy_loss + 0.5 * value_loss
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()


	def save_model(self, path):
		# Save the trained model to the specified path
		torch.save(self.policy.state_dict(), path)

	def load_model(self, path):
		# Load a trained model from the specified path
		self.policy.load_state_dict(torch.load(path))

def train_agent():
	env = RobotEnvironment()
	state_dim = env.num_joints * 2 + len(env.generate_voxel_grid().flatten())
	action_dim = env.num_joints
	agent = PPOAgent(state_dim, action_dim)

	print("training")

	# Training loop
	for episode in range(500):
		env.reset()
		states, actions, rewards, values = [], [], [], []
		episode_reward = 0

		for step in range(200):
			# print("episode {}, step {}".format(episode, step))
			# Get the current state
			state = env.get_state()
			# Select an action using the policy
			action = agent.select_action(state) * 10  # Scale up action magnitudes
			# Apply the action to the environment
			env.apply_action(action)

			# Calculate the reward (example: distance to a target)
			target_position = [0, 0, 0.5]
			end_effector_state = p.getLinkState(env.robotId, env.num_joints - 1)[0]
			distance_to_target = np.linalg.norm(np.array(target_position) - np.array(end_effector_state))
			reward = -distance_to_target * 10  # Amplify the reward signal
			episode_reward += reward

			# Get the value of the current state from the critic
			_, state_value = agent.policy(state)

			# Store transitions
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			values.append(state_value.item())  # Use critic's output

		# Train the agent after each episode
		agent.train(states, actions, rewards, values)
		print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
		print("actions: ", actions)

	# Save the trained model
	agent.save_model("trained_ppo_model.pth")
	env.disconnect()

def execute_trained_agent():
	# Initialize the simulation environment
	env = RobotEnvironment()
	state_dim = env.num_joints * 2  # State dimension includes positions and velocities of joints
	action_dim = env.num_joints  # Action dimension equals the number of joints
	agent = PPOAgent(state_dim, action_dim)

	# Load the pre-trained PPO model
	agent.load_model("trained_ppo_model.pth")

	# Define a set of feasible target positions for the end-effector
	target_positions = [
		[0.2, 0.1, 0.4],  # Pose near the middle of the first shelf
		[0.3, -0.2, 0.6],  # Pose near the second shelf
		[0.0, 0.0, 0.8],   # Pose near the third shelf
		[-0.2, 0.2, 1.0]   # Pose near the top shelf
	]

	# Iterate through each target position
	for target_position in target_positions:
		# Reset the environment for each target
		env.reset()
		print(f"Moving to target position: {target_position}")

		# Attempt to reach the target position within a step limit
		for step in range(200):
			# Get the current state of the robot
			state = env.get_state()
			# Use the policy to select an action based on the current state
			action = agent.select_action(state)
			# Apply the action to the robot
			env.apply_action(action)

			# Get the current position of the end-effector
			end_effector_state = p.getLinkState(env.robotId, env.num_joints - 1)[0]
			# Calculate the distance to the target position
			distance_to_target = np.linalg.norm(np.array(target_position) - np.array(end_effector_state))

			# Check if the end-effector is sufficiently close to the target
			if distance_to_target < 0.05:
				print(f"Reached target position: {target_position} in {step + 1} steps.")
				break
		else:
			print(f"Failed to reach target position: {target_position} within the step limit.")

	# Disconnect from the simulation
	env.disconnect()

if __name__ == "__main__":
	# Train the agent
	train_agent()
	# Execute the trained agent
	execute_trained_agent()