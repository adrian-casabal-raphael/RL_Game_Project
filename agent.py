import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import gym_tetris as tetris
import cv2
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace


def preprocess_state(state):
    print(f"original state shape: {state.shape}")
    if len(state.shape) == 3 and state.shape[2] == 3:
        gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) # converts to grayscale
    else:
        gray = state
    resized = cv2.resize(gray, (84, 84)) # resize to fixed state
    normalized = resized / 255.0 # normalize pixel values
    # add extra dimension for compatibility with NN
    preprocessed_state = np.expand_dims(normalized, axis=-1)
    print(f"Preprocessed state shape: {preprocessed_state.shape}")
    return preprocessed_state

# define architecture of neural network
def build_model(input_shape, action_size):
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='softmax')) # change activation to 'linear'
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# create DQN agent class
class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
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
        if len(self.memory) > 2000:
            self.memory.popleft()
    def choose_action(self, state, epsilon=1.0):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0) # subject to change
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

# training loop
def train(agent, env, episodes=10, batch_size=32):
    for episode in range(episodes):
        total_reward = 0
        done = True
        state = None
        for time in range(5000):
            if done:
                state = env.reset()
                state = preprocess_state(state)
                state = np.reshape(state, [1, *agent.input_shape])
                agent.update_target_model()

            action = agent.choose_action(state, epsilon=agent.epsilon)
            next_state, reward, done, info = env.step(action)
            env.render()
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, *agent.input_shape])
            # rewards
            lines_cleared = info.get('number_of_lines', 0)
            if lines_cleared == 1:
                reward += 10
            elif lines_cleared == 2:
                reward += 30
            elif lines_cleared == 3:
                reward += 60
            elif lines_cleared == 4:
                reward += 100

            # penalty for high stacking
            if np.any(state[:, 0:20]):
                reward -= 5

            # penalty for holes
            holes = np.sum((next_state == 0) & (np.cumsum(next_state, axis=0) > 0))
            reward -= 2 * holes

            # penalty for each time step
            reward -= 0.1

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        print(f"End of episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    env.close()



env = tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)
input_shape = (84, 84, 1)  # Shape after preprocessing
action_size = env.action_space.n
agent = DQNAgent(input_shape, action_size)

# Train the agent
train(agent, env)