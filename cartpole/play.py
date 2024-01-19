import os
import argparse
import json
import time
import numpy as np
import gymnasium as gym
from cartpole.agent import QLearningAgent

def main():
    """
    Main function to play cartpole given an agent.
    """
    
    t0 = time.time()
    
    # Initial print
    print("==========================================================================")
    print("==========================================================================")
    print("==================== CARTPOLE PLAYING SCRIPT STARTED ====================")
    print("==========================================================================")
    print("==========================================================================")
    
    # Argument parser for command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params_path", help="Path to json params dict")
    parser.add_argument("--agent_dir", type=str, default="./results/learn_run_1", help="Path to agent directory (save directory for cartpole_train tool).")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to play.")
    parser.add_argument("--save_dir", type=str, default="./results/play_run_1", help="Directory to save the results.")

    # Parse args
    args = parser.parse_args()
    agent, n_episodes, save_dir = parse_args(args)
    
    # Prepare save dir and paths
    os.makedirs(save_dir, exist_ok=True)
    save_gif_path = os.path.join(save_dir,"game.gif")
    save_rewards_plot_path = os.path.join(save_dir,"rewards.png")
    save_rewards_path = os.path.join(save_dir,"rewards.npy")
    save_frames_path = os.path.join(save_dir,"frames.npy")
    
    # Play cartpole
    rewards, frames = agent.play(n_episodes, save_gif_path)
    
    # Plot rewards
    agent.plot_rewards(rewards, save_rewards_plot_path)    
    
    # Save rewards and frames
    np.save(save_rewards_path, rewards)
    np.save(save_frames_path, frames)
    
    print("\n=== Script finished ===")
    print(f"   - It took {round((time.time()-t0)/60,2)} minutes to complete")
    print(f"   - Results on {save_dir}")
    
def parse_args(args):
    
    if args.params_path:
        with open(args.params_path, "r") as f:
            params = json.load(f)
        agent_dir, n_episodes, save_dir = params["agent_dir"], params["n_episodes"], params["save_dir"]
        
    else:
        agent_dir, n_episodes, save_dir = args.agent_dir, args.n_episodes, args.save_dir


    with open(os.path.join(agent_dir,"params.json"), "r") as f:
        agent_params = json.load(f)
    
    env = gym.make("CartPole-v1",render_mode="rgb_array")
    q_table = np.load(os.path.join(agent_dir,"q_table.npy"))
    agent = QLearningAgent(env,
                           learning_rate=agent_params["learning_rate"], 
                           discount_factor=agent_params["discount_factor"],
                           start_exploration_prob=agent_params["start_exploration_prob"],
                           final_exploration_prob=agent_params["final_exploration_prob"],
                           exploration_decay=agent_params["exploration_decay"],
                           max_values=agent_params["max_values"],
                           min_values=agent_params["min_values"],
                           num_bins=agent_params["num_bins"])
    agent.q_table = q_table
    
    return agent, n_episodes, save_dir
    