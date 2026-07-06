import math
import cv2
import time
import random
import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer():
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def push(self, transition):
        self.buffer.append(transition)

    def batch_sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Initialize the network
        super(Net, self).__init__()

        hidden_nodes1 = 512
        hidden_nodes2 = 256
        self.fc1 = nn.Linear(state_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, action_dim)

    def forward(self, state):
        # Define the forward pass of the actor
        x = state

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class Agent():
    def __init__(self,
                 env,
				 target_update_freq=1000,
                 learning_rate=3e-4,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.999954,
                 gamma=0.99,
                 batch_size=25,
                 max_grad_norm=1.0,
                 total_episodes=100000):

        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_update_freq = target_update_freq
        self.total_episodes = total_episodes
        self.total_steps = 0

        self.q_net = Net(state_dim=25, action_dim=env.action_space.n).to(device)
        self.target_net = Net(state_dim=25, action_dim=env.action_space.n).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(size=5000)

    def complete_episode(self, episode):
        state, _ = self.env.reset()
        obs = state
        done = False
        episode_reward = 0
        speeds = []

        while not done:
            state_proc = state.flatten()
            state_proc = torch.tensor(state_proc, dtype=torch.float32, device=device)

            speed = state.flatten()[3] * 20
            speeds.append(speed)

            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample() # Take random action
            else:
                with torch.no_grad():
                    action = self.q_net(state_proc).argmax().item() # Take action with highest expected reward

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            obs = next_state
            done = terminated or truncated

            next_state_proc = next_state.flatten()

            self.buffer.push((state_proc.cpu().numpy(), action, reward, next_state_proc, done)) # Store transition

            episode_reward += reward
            self.total_steps += 1
            state = next_state

            if len(self.buffer) >= self.target_update_freq:
                loss = self.update_network()

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # Decay epsilon

        return episode_reward, sum(speeds) / len(speeds), terminated

    def update_network(self):
        states, next_states, actions, rewards, dones = self.buffer.batch_sample(self.batch_size)

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_vals = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_vals * (1 - dones)

        # Optimization
        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm) # Gradient clipping to ensure updates aren't too large

        self.optimizer.step()
        return loss

    def train(self):
        for episode in range (self.total_episodes):
            episode_reward, average_speed, terminated = self.complete_episode(episode)

            # TensorBoard logging
            writer.add_scalar("charts/episode_reward", episode_reward, episode)
            writer.add_scalar("charts/epsilon", self.epsilon, episode)
            writer.add_scalar("charts/average_speed", average_speed, episode)
            if terminated: # Agent crashed
                writer.add_scalar("charts/crashed", 1, episode)
            else: # Agent didn't crash
                writer.add_scalar("charts/crashed", 0, episode)
            writer.flush()

            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {self.epsilon:.3f}")

# Environment setup
env = gym.make('highway-fast-v0', render_mode=None)
currentTime = time.time()
run_name = f"DQN_Highway_{int(currentTime)}_speed_reward"
writer = SummaryWriter(f"runs/{run_name}")

agent = Agent(env)
agent.train()
