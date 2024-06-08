import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import gymnasium as gym
import gym_tetris as tetris
import cv2
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import Wrapper

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
    if len(state.shape) == 3 and state.shape[2] == 3:
        gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) # converts to grayscale
    else:
        gray = state
    resized = cv2.resize(gray, (84, 84)) # resize to fixed state
    normalized = resized / 255.0 # normalize pixel values
    # add extra dimension for compatibility with NN
    preprocessed_state = np.expand_dims(normalized, axis=-1)
    return preprocessed_state

# define architecture of neural network
def build_model(input_shape, action_size):
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model



# create DQN agent class
class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(input_shape, action_size)
        self.target_model = build_model(input_shape, action_size)
        self.update_target_model()
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def choose_action(self, state, epsilon=1.0):
        state = np.expand_dims(state, axis=0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.vstack(states)
        states = np.reshape(states, (batch_size, *agent.input_shape))
        next_states = np.vstack(next_states)
        next_states = np.reshape(next_states, (batch_size, *agent.input_shape))
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(batch_size):
            target = rewards[i] if dones[i] else rewards[i] + self.gamma * np.amax(next_q_values[i])
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

# Custom gym wrapper to remove seed argument
class CustomEnvWrapper(Wrapper):
    def reset(self, **kwargs):
        if 'seed' in kwargs:
            kwargs.pop('seed')
        obs = self.env.reset(**kwargs)
        if not isinstance(obs, tuple):
            obs = (obs, {})
        return obs
    def step(self, action):
        results = self.env.step(action)
        if len(results) == 4:
            obs, reward, done, info = results
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = results
        return obs, reward, terminated, truncated, info

# training loop
def train(agent, env, episodes=1000, batch_size=32):
    num_envs = len(env.remotes)
    for episode in range(episodes):
        results = env.reset()
        if isinstance(results, tuple):
            states, infos = results
        else:
            states = results
            infos = [{} for _ in range(num_envs)]
        states = np.array([preprocess_state(state) for state in states])
        states = np.reshape(states, (num_envs, *agent.input_shape))
        total_rewards = np.zeros(num_envs)
        for time in range(1000):
            print(f"in time step: {time}")
            actions = [agent.choose_action(state, epsilon=agent.epsilon) for state in states]
            next_states, rewards, dones, infos = env.step(actions)
            next_states = np.array([preprocess_state(state) for state in next_states])
            next_states = np.reshape(next_states, (num_envs, *agent.input_shape))

            # rewards
            for i in range(num_envs):
                lines_cleared = infos[i].get('number_of_lines', 0)
                if lines_cleared == 1:
                    rewards[i] += 10
                elif lines_cleared == 2:
                    rewards[i] += 30
                elif lines_cleared == 3:
                    rewards[i] += 60
                elif lines_cleared == 4:
                    rewards[i] += 100

                # penalty for high stacking
                if np.any(next_states[i][:, 0:20]):
                    rewards[i] -= 5

                # penalty for holes
                holes = np.sum((next_states[i] == 0) & (np.cumsum(next_states[i], axis=0) > 0))
                rewards[i] -= 2 * holes

                # penalty for each time step
                rewards[i] -= 0.1

                agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                total_rewards[i] += rewards[i]

                if dones[i]:
                    states[i], infos[i] = env.reset()
                    states[i] = preprocess_state(states[i])
                    states[i] = np.reshape(states[i], agent.input_shape)
                    agent.update_target_model()
                else:
                    states[i] = next_states[i]

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        print(f"End of episode {episode}, Total Reward: {total_rewards}, Epsilon: {agent.epsilon}")
    env.close()

def make_env(env_id, rank, seed=0):
    def _init():
        env = tetris.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = CustomEnvWrapper(env) # to remove seed
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == '__main__':
    num_envs = 4 # number of parallel environments
    env_id = 'TetrisA-v3'
    envs = [make_env(env_id, i) for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    input_shape = (84, 84, 1)  # Shape after preprocessing
    action_size = env.action_space.n
    agent = DQNAgent(input_shape, action_size)

    # Train the agent
    train(agent, env)