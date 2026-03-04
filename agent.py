import os
import random
from collections import deque

import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from src.tetris import Tetris

os.makedirs("recordings", exist_ok=True)
os.makedirs("models", exist_ok=True)


def build_action_space(env):
    actions = set()
    for piece in env.pieces:
        rotated_piece = [row[:] for row in piece]
        seen_shapes = set()
        for rotation in range(4):
            shape_key = tuple(tuple(row) for row in rotated_piece)
            if shape_key in seen_shapes:
                rotated_piece = env.rotate(rotated_piece)
                continue
            seen_shapes.add(shape_key)
            max_x = env.width - len(rotated_piece[0])
            for x in range(max_x + 1):
                actions.add((x, rotation))
            rotated_piece = env.rotate(rotated_piece)
    return sorted(actions)


def build_action_mask(valid_actions, action_to_index, action_dim):
    mask = torch.zeros(action_dim, dtype=torch.bool)
    for action in valid_actions:
        idx = action_to_index.get(action)
        if idx is not None:
            mask[idx] = True
    return mask


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.feature(x)
        advantages = self.advantage(features)
        value = self.value(features)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid_mask):
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_valid_mask = zip(
            *random.sample(self.buffer, batch_size)
        )
        return state, action, reward, next_state, done, next_valid_mask

    def __len__(self):
        return len(self.buffer)


def train(model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done, next_valid_mask = replay_buffer.sample(batch_size)

    state = torch.stack(state).float()
    action = torch.tensor(action, dtype=torch.long).unsqueeze(1)
    reward = torch.tensor(reward, dtype=torch.float32)
    next_state = torch.stack(next_state).float()
    done = torch.tensor(done, dtype=torch.float32)
    next_valid_mask = torch.stack(next_valid_mask).bool()

    q_values = model(state)
    q_value = q_values.gather(1, action).squeeze(1)

    next_q_values = model(next_state)
    min_value = torch.finfo(next_q_values.dtype).min
    masked_next_q_values = next_q_values.masked_fill(~next_valid_mask, min_value)
    next_q_value = masked_next_q_values.max(1)[0]
    has_valid_actions = next_valid_mask.any(dim=1)
    next_q_value = torch.where(has_valid_actions, next_q_value, torch.zeros_like(next_q_value))
    next_q_value = torch.where(done.bool(), torch.zeros_like(next_q_value), next_q_value)

    expected_q_value = reward + gamma * next_q_value
    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model, episode):
    model_path = f"models/tetris_model_{episode + 1}.pth"
    torch.save(model.state_dict(), model_path)


def load_model(model, episode):
    model_path = f"models/tetris_model_{episode}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return True
    print(f"No model found at {model_path}")
    return False


def main():
    env = Tetris()
    action_space = build_action_space(env)
    action_to_index = {action: idx for idx, action in enumerate(action_space)}
    index_to_action = {idx: action for idx, action in enumerate(action_space)}

    model = DQN(input_dim=4, output_dim=len(action_space))
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(10000)
    num_episodes = 3000
    batch_size = 32
    gamma = 0.99

    model_loaded = load_model(model, 3000)
    epsilon = 0.1 if model_loaded else 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.1

    for episode in range(num_episodes):
        state = env.reset().flatten().float()
        total_reward = 0

        video_path = f"recordings/episode_{episode}.avi"
        frame_width = env.width * env.block_size + env.extra_board.shape[1]
        frame_height = env.height * env.block_size
        video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            10,
            (frame_width, frame_height),
        )
        if not video.isOpened():
            # Fallback codec/container for environments where XVID is unavailable.
            video_path = f"recordings/episode_{episode}.mp4"
            video = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (frame_width, frame_height),
            )

        while True:
            next_states = env.get_next_states()
            valid_actions = list(next_states.keys())
            if not valid_actions:
                break
            valid_indices = [action_to_index[action] for action in valid_actions]

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = model(state.unsqueeze(0))
                    best_index = max(valid_indices, key=lambda idx: q_values[0, idx].item())
                    action = index_to_action[best_index]

            reward, done = env.step(action, render=True, video=video)
            next_state = next_states[action].flatten().float()

            if done:
                reward -= 10
                next_valid_mask = torch.zeros(len(action_space), dtype=torch.bool)
            else:
                reward += 10 * env.cleared_lines
                reward -= env.get_holes(env.board)
                next_valid_actions = list(env.get_next_states().keys())
                next_valid_mask = build_action_mask(
                    next_valid_actions,
                    action_to_index,
                    len(action_space),
                )

            replay_buffer.push(
                state,
                action_to_index[action],
                reward,
                next_state,
                done,
                next_valid_mask,
            )

            state = next_state
            total_reward += reward

            train(model, optimizer, replay_buffer, batch_size, gamma)

            if done:
                break

        video.release()
        cv2.destroyAllWindows()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")
        save_model(model, episode)


if __name__ == "__main__":
    main()
