import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import cv2
import os
import hashlib
from src.tetris import Tetris
import torch as torch

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def preprocess_state(state):
    if isinstance(state, torch.Tensor):
        state = state.numpy()

    if state.dtype != np.uint8:
        state = state.astype(np.uint8)

    resized = cv2.resize(state, (84, 84)) / 255.0  # Normalize pixel values
    return np.expand_dims(resized, axis=-1)  # Add channel dimension


# define architecture of neural network
def build_model(input_shape, feature_shape, action_size):
    grid_input = Input(shape=input_shape)
    feature_input = Input(shape=feature_shape)

    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(grid_input)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    concatenated = Concatenate()([x, feature_input])
    y = Dense(512, activation='relu')(concatenated)
    output = Dense(action_size, activation='linear')(y)

    model = tf.keras.Model(inputs=[grid_input, feature_input], outputs=output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))
    return model


# class to normalize rewards
class RewardNormalizer:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 0

    def normalize(self, reward):
        self.count += 1
        alpha = 1.0 / self.count
        self.mean = (1 - alpha) * self.mean + alpha * reward
        self.var = (1 - alpha) * self.var + alpha * (reward - self.mean) ** 2
        std = np.sqrt(self.var)
        normalized_reward = (reward - self.mean) / (std + 1e-8)
        return normalized_reward

def get_action_space(env):
    action_space = []
    for x in range(env.width):
        for rotations in range(4):  # Assuming up to 4 possible rotations
            action_space.append((x, rotations))
    return action_space


# create DQN agent class
class DQNAgent:
    def __init__(self, env, input_shape, feature_shape, action_size, initial_epsilon=1.0):
        self.env = env
        self.input_shape = input_shape
        self.action_size = action_size
        self.feature_shape = feature_shape
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = initial_epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(input_shape, feature_shape, action_size)
        self.target_model = build_model(input_shape, feature_shape, action_size)
        self.update_target_model()
        self.reward_normalizer = RewardNormalizer()
        self.visited_states = set()
        self.action_space = get_action_space(self.env)
    def update_target_model(self, tau=0.1):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = []
        for model_weight, target_weight in zip(model_weights, target_weights):
            new_weight = tau * model_weight + (1 - tau) * target_weight
            new_weights.append(new_weight)
        self.target_model.set_weights(new_weights)
    def remember(self, state, features, action, reward, next_state, next_features, done):
        normalized_reward = self.reward_normalizer.normalize(reward)
        self.memory.append((state, features, action, normalized_reward, next_state, next_features, done))
    def choose_action(self, state, features, epsilon):
        if np.random.rand() <= epsilon:
            return random.choice(self.action_space)
        act_values = self.model.predict([state, features], verbose=0)
        action_index = np.argmax(act_values[0])
        action = self.action_space[action_index]

        # Validate action
        x, num_rotations = action
        valid_x = x >= 0 and x < self.env.width
        valid_rotations = num_rotations >= 0 and num_rotations < 4

        if not (valid_x and valid_rotations):
            return random.choice(self.action_space)

        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([m[0] for m in minibatch])
        features = np.vstack([m[1] for m in minibatch])
        actions = np.array([m[2] for m in minibatch])
        rewards = np.array([m[3] for m in minibatch])
        next_states = np.vstack([m[4] for m in minibatch])
        next_features = np.vstack([m[5] for m in minibatch])
        dones = np.array([m[6] for m in minibatch])

        targets = self.model.predict([states, features], verbose=0)
        next_q_values = self.target_model.predict([next_states, next_features], verbose=0)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                max_expected = np.max(next_q_values[i])
                targets[i][actions[i]] = rewards[i] + self.gamma * max_expected
        self.model.fit([states, features], targets, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)
    def load_target(self, name):
        self.target_model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)
    def save_target(self, name):
        self.target_model.save_weights(name)

def get_latest_model(directory):
    if not os.path.exists(directory):
        return None
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    return os.path.join(directory, latest_model)

def count_holes(grid):
    holes = 0
    for col in range(grid.shape[1]):
        block_found = False
        for row in range(grid.shape[0]):
            if grid[row,col] != 0:
                block_found = True
            elif block_found and grid[row,col] == 0:
                holes += 1;
    return holes

# function to store already visited states
def state_to_hashable(state):
    return hashlib.sha256(state.tobytes()).hexdigest()

# training loop
def train(agent, env, episodes=1501, batch_size=64, render_freq=100, record=False, output_dir='recordings', model_dir='models', target_dir ='target_models', max_steps=5000):
    episode_rewards = []
    if record and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for episode in range(episodes):
        if (episode + 1) % 500 == 0:
            batch_size = min(batch_size * 2, 512)
        if (episode + 1) % 500 == 0:
            max_steps = max_steps * 2

        if record and episode % render_freq == 0:
            state = env.reset()
            frame = env.render()
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = os.path.join(output_dir, f'agent_playing_episode_{episode}.avi')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width,height))

        state = env.reset()
        state_img = env.get_current_board_state()
        state_img = preprocess_state(state_img)
        state_img = np.reshape(state_img, [1, *agent.input_shape])

        state_features = env.get_state_properties(env.board)
        if isinstance(state_features, torch.Tensor):
            state_features = state_features.numpy()
        state_features = np.reshape(state_features, [1, *agent.feature_shape])
        state_hash = state_to_hashable(state_img)

        total_reward = 0
        done = False
        for time in range(max_steps):
            if time % 1000 == 0:
                print(f"in time step: {time}")
            if time % 10 == 0 and episode % render_freq == 0:
                frame = env.render()
                if record and (frame is not None):
                    out.write(frame)
                cv2.imshow('Tetris', frame)
                cv2.waitKey(1)

            action = agent.choose_action(state_img, state_features, agent.epsilon)
            reward, done = env.step(action)
            next_state_img = env.get_current_board_state()
            next_state_img = preprocess_state(next_state_img)
            next_state_img = np.reshape(next_state_img, [1, *agent.input_shape])

            next_state_features = env.get_state_properties(env.board)
            if isinstance(next_state_features, torch.Tensor):
                next_state_features = next_state_features.numpy()
            next_state_features = np.reshape(next_state_features, [1, *agent.feature_shape])
            next_state_hash = state_to_hashable(next_state_img)

            if done:
                reward -= 100  # significant penalty for resetting game due to stacking
                agent.remember(state_img, state_features, action, reward, next_state_img, next_state_features, done)
                total_reward += reward
                agent.replay(batch_size)
                break

            agent.remember(state_img, state_features, action, reward, next_state_img, next_state_features, done)
            state_img = next_state_img
            state_features = next_state_features
            state_hash = next_state_hash
            total_reward += reward

            if (time + 1) % 500 == 0:
                agent.replay(batch_size)

        if (episode + 1) % 10 == 0:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
        if (episode + 1) % 100 == 0:
            agent.update_target_model()

        print(f"End of episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        if episode % 50 == 0:
            model_path = os.path.join(model_dir, f"model_checkpoint_{episode}.h5")
            agent.save(model_path)
            target_model_path = os.path.join(target_dir, f"model_checkpoint_{episode}.h5")
            agent.save_target(target_model_path)
        if record:
            out.release()
            cv2.destroyAllWindows()


    # save the final model
    agent.save(os.path.join(model_dir, "final_model.h5"))
    agent.save_target(os.path.join(target_dir, "final_model.h5"))



# set up environment
env = Tetris()
input_shape = (84, 84, 1)  # Shape after preprocessing
feature_shape = (4,)
action_size = len(get_action_space(env))

# load model if available
model_dir = 'models'
target_dir = 'target_models'
latest_model_path = get_latest_model(model_dir)
latest_target_path = get_latest_model(target_dir)
if latest_model_path and latest_target_path:
    agent = DQNAgent(env, input_shape, feature_shape, action_size, initial_epsilon=0.4)
    agent.load(latest_model_path)
    agent.load_target(latest_target_path)
    print(f"loaded model from {latest_model_path} and {latest_target_path}")
else:
    agent = DQNAgent(env, input_shape, feature_shape, action_size)

# Train the agent
train(agent, env, record=True)