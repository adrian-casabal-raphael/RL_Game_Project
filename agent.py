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

# Ensure directories exist
os.makedirs("recordings", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define the neural network for the Q-learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
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


# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Training the DQN
def train(env, model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.stack(state)
    action = torch.tensor(action, dtype=torch.long).unsqueeze(1)
    reward = torch.tensor(reward)
    next_state = torch.stack(next_state)
    done = torch.tensor(done, dtype=torch.float32)

    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the model
def save_model(model, episode):
    model_path = f"models/tetris_model_{episode + 1}.pth"
    torch.save(model.state_dict(), model_path)

# Load the model and return a flag indicating success
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

# Main function to train the DQN agent
def main():
    env = Tetris()
    possible_actions = list(env.get_next_states().keys())
    action_to_index = {action: idx for idx, action in enumerate(possible_actions)}
    index_to_action = {idx: action for idx, action in enumerate(possible_actions)}

    model = DQN(input_dim=4, output_dim=len(possible_actions))  # 4 features and number of possible actions
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(10000)
    num_episodes = 3000
    batch_size = 32
    gamma = 0.99

    # Load the model if it exists and set a flag
    model_loaded = load_model(model, 3000)  # Change the episode number as needed

    # Set epsilon based on whether the model was loaded successfully
    if model_loaded:
        print(f"model {model_loaded} loaded")
        epsilon = 0.1  # Specific epsilon for loaded model
    else:
        epsilon = 1.0
    epsilon_decay = 0.999  # Slower decay
    epsilon_min = 0.1  # Ensure some exploration even at the end

    for episode in range(num_episodes):
        state = env.reset()
        state = state.flatten()
        total_reward = 0

        video_path = f"recordings/episode_{episode}.avi"
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10,
                                (env.width * env.block_size * 2, env.height * env.block_size))

        while True:
            if random.random() < epsilon:
                # exploration: choose a random action
                action = random.choice(possible_actions)
            else:
                with torch.no_grad():
                    # exploitation: choose the greedy action
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                    action_index = torch.argmax(q_values).item()
                    action = index_to_action[action_index]

            next_states = env.get_next_states()

            # Ensure the action is valid
            if action not in next_states:
                continue

            reward, done = env.step(action, render=True, video=video)
            next_state = next_states[action].flatten()

            # Modify reward shaping
            if done:
                reward -= 10  # Penalize game over
            else:
                reward += 10 * env.cleared_lines
                reward -= env.get_holes(env.board)

            action_index = action_to_index[action]

            replay_buffer.push(
                torch.FloatTensor(state),
                action_index,  # Store action index
                reward,
                torch.FloatTensor(next_state),
                done
            )

            state = next_state
            total_reward += reward

            train(env, model, optimizer, replay_buffer, batch_size, gamma)

            if done:
                break

        video.release()
        cv2.destroyAllWindows()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

        save_model(model, episode)

if __name__ == "__main__":
    main()
