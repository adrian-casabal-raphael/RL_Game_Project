import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import cv2
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from tetris import Tetris

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
    state = np.array(state)
    print(f"Original state shape: {state.shape}")
    if len(state.shape) == 3 and state.shape[2] == 3:
        gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) # converts to grayscale
    else:
        gray = state
    print(f"Grayscale state shape: {gray.shape}")
    if gray.shape[0] >= 1 and gray.shape[1] > 0:
        resized = cv2.resize(gray, (84, 84)) # resize to fixed state
    else:
        resized = np.zeros((84, 84)) # default to zero if resize fails
    print(f"Resized state shape: {resized.shape}")
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
        states = np.reshape(states, (batch_size, *self.input_shape))
        next_states = np.vstack(next_states)
        next_states = np.reshape(next_states, (batch_size, *self.input_shape))
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(batch_size):
            target = rewards[i] if dones[i] else rewards[i] + self.gamma * np.amax(next_q_values[i])
            targets[i][actions[i]] = target.item()

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

# training loop
def train(agent, env, episodes=1000, batch_size=32, render=True):
    #render_env = env.envs[0]
    episode_rewards = []
    video_writer = None
    for episode in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.reshape(state, agent.input_shape)
        total_reward = 0


        if render and episode % 100 == 0:
            video_path = f"recordings/episode_{episode}.avi"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (env.grid_width * 30, env.grid_height * 30)) # NES Resolution

        for time in range(1000):
            print(f"in time step: {time}")
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step([action])
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, agent.input_shape)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if render:
                env.render(video=video_writer)

            if render and episode % 100 == 0:
                frame = env.render(video=video_writer)
                if frame is not None and frame.size > 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_writer.write(frame)

            if done:
                agent.update_target_model()
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if video_writer:
            video_writer.release()
            video_writer = None

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        episode_rewards.append(total_reward)
        print(f"End of episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    env.close()

    # plotting the rewards
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.show()



if __name__ == '__main__':
    env = Tetris()
    #env = DummyVecEnv([lambda: env])
    input_shape = (84, 84, 10, 1)
    action_size = env.action_space.n
    agent = DQNAgent(input_shape, action_size)

    # Train the agent
    train(agent, env, render=True)