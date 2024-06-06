from agent import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def eval_agent(env, agent, episodes=10):
    total_rewards = []
    total_lines_cleared = []
    for e in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.reshape(state, [1, *agent.input_shape])
        done = False
        total_reward = 0
        lines_cleared = 0

        while not done:
            action = agent.choose_action(state, epsilon=0)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            print(f"Raw next_state shape: {next_state.shape}")
            next_state = preprocess_state(next_state)
            print(f"Preprocessed next_state shape: {next_state.shape}")
            next_state = np.reshape(next_state, [1, *agent.input_shape])
            print(f"Reshaped next_state shape: {next_state.shape}")
            state = next_state
            total_reward += reward
            lines_cleared += info.get('lines_cleared', 0)
        total_rewards.append(total_reward)
        total_lines_cleared.append(lines_cleared)
        print(f"Episode: {e + 1}/{episodes}, Reward: {total_reward}, Lines Cleared: {lines_cleared}")

    avg_reward = np.mean(total_rewards)
    avg_lines_cleared = np.mean(total_lines_cleared)
    print(f"Average Reward over {episodes}: {avg_reward}")
    print(f"Average Lines Cleared over {episodes}: {avg_lines_cleared}")

    return avg_reward, avg_lines_cleared

def tune_hyperparams(agent, env, episodes=1000, batch_size=32):
    learning_rates = [0.001, 0.0005, 0.0001]
    gamma_values = [0.95, 0.99]
    epsilon_decays = [0.995, 0.99, 0.9]
    batch_sizes = [32, 64, 128]

    best_avg_reward = -float('inf')
    best_params = {}

    for lr in learning_rates:
        for gamma in gamma_values:
            for epsilon_decay in epsilon_decays:
                for batch_size in batch_sizes:
                    agent.learning_rate = lr
                    agent.gamma = gamma
                    agent.epsilon_decay = epsilon_decay
                    agent.model.optimizer.learning_rate = lr
                    train(agent, env, episodes, batch_size)

                    avg_reward, avg_lines_cleared = eval_agent(env, agent, episodes=10)

                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_params = {
                            'learning_rate': lr,
                            'gamma': gamma,
                            'epsilon_decay': epsilon_decay,
                            'batch_size': batch_size}
    print(f"Best Avg Reward: {best_avg_reward}")
    print(f"Best Hyperparameters: {best_params}")

    agent.learning_rate = best_params['learning_rate']
    agent.gamma = best_params['gamma']
    agent.epsilon_decay = best_params['epsilon_decay']
    agent.model.optimizer.learning_rate = best_params['learning_rate']

def visualize_training_progress(rewards, losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1,2,2)
    plt.plot(losses)
    plt.title("Loss over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")

def train(agent, env, episodes=1000, batch_size=32):
    rewards = []
    losses = []
    writer = SummaryWriter(log_dir='runs/DQN_Tetris')

    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        state = preprocess_state(state)
        state = np.reshape(state, [1, *agent.input_shape])
        for time in range(500):
            action = agent.choose_action(state, epsilon=agent.epsilon)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, *agent.input_shape])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_model()
                print(f"episode: {episode}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        rewards.append(total_reward)
        writer.add_scalar("Total Reward", total_reward, episode)

        if losses:
            writer.add_scalar("loss", losses[-1], episode)

        visualize_training_progress(rewards, losses)
        writer.close()



def render_agent_play(agent, env, episodes=5):
    for e in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.reshape(state, [1, *agent.input_shape])
        done = False

        while not done:
            env.render()
            action = agent.choose_action(state, epsilon=0)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, *agent.input_shape])
            state = next_state
        env.close()

if __name__ == '__main__':
    env = gym.make('TetrisB-v3')
    input_shape = (84, 84, 1)
    action_size = env.action_space.n
    agent = DQNAgent(input_shape=input_shape, action_size=action_size)

    avg_reward, avg_lines_cleared = eval_agent(env, agent, episodes=10)

    tune_hyperparams(agent, env, episodes=1000, batch_size=32)
    train(agent, env, episodes=1000, batch_size=32)
    render_agent_play(agent, env, episodes=5)