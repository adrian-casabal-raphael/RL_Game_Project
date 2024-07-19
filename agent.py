import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
from collections import deque
from PIL import Image
from src.tetris import Tetris
import logging
import time

# Ensure directories exist
os.makedirs("recordings", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(level=logging.DEBUG)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature(x)
        advantages = self.advantage(features)
        value = self.value(features)
        return value + (advantages - advantages.mean())

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], [], [], [], []

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        state, action, reward, next_state, done = zip(*samples)
        return state, action, reward, next_state, done, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

def train(env, model, target_model, optimizer, replay_buffer, batch_size, gamma, beta):
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done, weights, indices = replay_buffer.sample(batch_size, beta)

    state = torch.stack(state).to(device)
    action = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(device)
    reward = torch.tensor(reward).to(device)
    next_state = torch.stack(next_state).to(device)
    done = torch.tensor(done, dtype=torch.float32).to(device)

    q_values = model(state)
    next_q_values = model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.argmax(next_q_values, 1).unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

def save_model(model, episode):
    model_path = f"models/tetris_model_{episode + 1}.pth"
    torch.save(model.state_dict(), model_path)

def load_model(model, episode):
    model_path = f"models/tetris_model_{episode}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return True
    else:
        print(f"No model found at {model_path}")
        return False

def main():
    env = Tetris()
    possible_actions = list(env.get_next_states().keys())
    action_to_index = {action: idx for idx, action in enumerate(possible_actions)}
    index_to_action = {idx: action for idx, action in enumerate(possible_actions)}

    model = DuelingDQN(input_dim=4, output_dim=len(possible_actions)).to(device)
    target_model = DuelingDQN(input_dim=4, output_dim=len(possible_actions)).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters())
    replay_buffer = PrioritizedReplayBuffer(100000)
    num_episodes = 20000
    batch_size = 512
    gamma = 0.99
    beta = 0.4
    beta_increment_per_sampling = 0.001

    model_loaded = load_model(model, 5000)

    if model_loaded:
        print(f"model {model_loaded} loaded")
        epsilon = 0.778
        beta = 1.0
    else:
        epsilon = 1.0

    epsilon_decay_interval = 100
    epsilon_decay = 0.995
    epsilon_min = 0.1

    for episode in range(num_episodes):
        state = env.reset()
        state = state.flatten()
        total_reward = 0

        video_path = f"recordings/episode_{episode}.avi"
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10,
                                (env.width * env.block_size * 2, env.height * env.block_size))

        logging.debug(f"Starting episode {episode + 1}")

        episode_start_time = time.time()  # Start timing the episode

        while True:
            if time.time() - episode_start_time > 300:  # Timeout after 300 seconds
                logging.warning(f"Episode {episode + 1} taking too long, terminating early.")
                break

            if random.random() < epsilon:
                action = random.choice(possible_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action_index = torch.argmax(q_values).item()
                    action = index_to_action[action_index]

            next_states = env.get_next_states()

            if action not in next_states:
                continue

            try:
                reward, done = env.step(action, render=True, video=video)
            except Exception as e:
                logging.error(f"Exception during env.step: {e}")
                break

            next_state = next_states[action].flatten()

            if done:
                reward -= 10
            else:
                reward += 10 * env.cleared_lines
                reward -= env.get_holes(env.board)

            action_index = action_to_index[action]

            replay_buffer.push(
                torch.FloatTensor(state).to(device),
                action_index,
                reward,
                torch.FloatTensor(next_state).to(device),
                done
            )

            state = next_state
            total_reward += reward

            train(env, model, target_model, optimizer, replay_buffer, batch_size, gamma, beta)

            if done:
                break

        if video:
            video.release()
        cv2.destroyAllWindows()

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        if (episode + 1) % epsilon_decay_interval == 0:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

        beta = min(1.0, beta + beta_increment_per_sampling)
        logging.debug(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}, Beta: {beta}")

        if (episode + 1) % 500 == 0:
            save_model(model, episode)

if __name__ == "__main__":
    main()
