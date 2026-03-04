import argparse
import os

import torch

from agent import DQN, build_action_space
from src.tetris import Tetris


def evaluate(model, env, action_to_index, index_to_action, episodes=10, render=False):
    total_scores = []
    total_lines = []

    for episode in range(episodes):
        state = env.reset().flatten().float()
        episode_score = 0

        while True:
            next_states = env.get_next_states()
            valid_actions = list(next_states.keys())
            if not valid_actions:
                break

            valid_indices = [action_to_index[action] for action in valid_actions]
            with torch.no_grad():
                q_values = model(state.unsqueeze(0))
                best_index = max(valid_indices, key=lambda idx: q_values[0, idx].item())
                action = index_to_action[best_index]

            reward, done = env.step(action, render=render)
            episode_score += reward
            state = next_states[action].flatten().float()

            if done:
                break

        total_scores.append(episode_score)
        total_lines.append(env.cleared_lines)
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Score: {episode_score} | Lines cleared: {env.cleared_lines}"
        )

    avg_score = sum(total_scores) / max(len(total_scores), 1)
    avg_lines = sum(total_lines) / max(len(total_lines), 1)
    print(f"Average score: {avg_score:.2f}")
    print(f"Average lines cleared: {avg_lines:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN Tetris agent.")
    parser.add_argument(
        "--model-episode",
        type=int,
        default=3000,
        help="Episode number used in model filename models/tetris_model_<episode>.pth",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render gameplay while evaluating")
    return parser.parse_args()


def main():
    args = parse_args()

    env = Tetris()
    action_space = build_action_space(env)
    action_to_index = {action: idx for idx, action in enumerate(action_space)}
    index_to_action = {idx: action for idx, action in enumerate(action_space)}

    model = DQN(input_dim=4, output_dim=len(action_space))

    model_path = f"models/tetris_model_{args.model_episode}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    evaluate(
        model,
        env,
        action_to_index,
        index_to_action,
        episodes=args.episodes,
        render=args.render,
    )


if __name__ == "__main__":
    main()
