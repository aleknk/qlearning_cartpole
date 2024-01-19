import os
import time
import argparse
import json
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from cartpole.agent import QLearningAgent
from cartpole.agent import QLearningAgent

def q_learning(agent: QLearningAgent, n_episodes:int=1000, show_every:int=100, verbose:int=1):
    """
    Run the Q-Learning algorithm to train the agent.
    
    Parameters:
    - agent: An instance of the QLearningAgent class.
    - n_episodes: Number of episodes to run (default is 1000).
    - show_every: Frequency of logging during training (default is 100).
    - verbose: Level of verbosity for logging (default is 1).
    
    Returns:
    - rewards: List of rewards received per episode.
    """
    
    # Initialization: print the loop starting if verbose is enabled.
    if verbose:
        print("\n=== Q-Learning Loop ===")

    # Initialize variables
    rewards = []  # List to store the total rewards for each episode
    
    # Loop through all episodes
    with tqdm(total=n_episodes, desc="Training Progress") as pbar:
        for episode in range(n_episodes):
                    
            if episode > 0:
                agent.decay_exploration(episode)
            
            # Initialize or reset state at the beginning of each episode
            state = agent.discretize_state(agent.env.reset()[0])
            episode_reward = 0  # To keep track of the total reward in the current episode

            # Loop through each step in the current episode
            while True:
                
                # Agent chooses an action
                action = agent.choose_action(state)
                
                # Environment returns next state and reward after action taken
                next_state, reward, terminated, truncated, _ = agent.env.step(action)
                next_state = agent.discretize_state(next_state)
                
                # Update the Q-values in Q-table
                agent.update(state, action, reward, next_state)

                # Prepare for the next iteration
                state = next_state
                episode_reward += reward

                # Break the loop if the episode is terminated or truncated
                if terminated or truncated:
                    break
                    
            # Append the total episode reward to the rewards list
            rewards.append(episode_reward)
            
            # Update progress bar
            pbar.set_postfix(reward=episode_reward, epsilon=agent.exploration_prob)
            pbar.update(1)
            
    return rewards


def main():
    """
    Main function to set up and run the Q-learning algorithm.
    """
    
    t0 = time.time()
    
    # Argument parsing.
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--params_path", help="Path to json params dict")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for Q-learning.")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor for Q-learning.")
    parser.add_argument("--start_exploration_prob", type=float, default=1.0, help="Initial exploration probability.")
    parser.add_argument("--final_exploration_prob", type=float, default=0.1, help="Maximum exploration probability.")
    parser.add_argument("--exploration_decay", type=float, default=1e6, help="Rate of decay for exploration probability.")
    parser.add_argument("--max_values", nargs='+', type=float, default=[2.4, 3.0, 0.3, 3.0], help="List of maximum values for state bins.")
    parser.add_argument("--min_values", nargs='+', type=float, default=[-2.4, -3.0, -0.3, -3.0], help="List of minimum values for state bins.")
    parser.add_argument("--num_bins", nargs='+', type=int, default=[20, 20, 20, 20], help="Number of bins for discretization of each state dimension.")
    parser.add_argument("--n_episodes", type=int, default=20000, help="Number of episodes for training.")
    parser.add_argument("--save_dir", type=str, default="./results/learn_run_1", help="Directory to save the results.")

    args = parser.parse_args()

    # Extract individual parameters.
    learning_rate, discount_factor, start_exploration_prob, final_exploration_prob, exploration_decay, max_values, min_values, num_bins, n_episodes, save_dir = parse_args(args)
    
    # Initial print
    print("===================================================================")
    print("===================================================================")
    print("==================== Q-LEARNING SCRIPT STARTED ====================")
    print("===================================================================")
    print("===================================================================")
    
    # Set up the environment.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # Instantiate Agent
    agent = QLearningAgent(env,
                           learning_rate=learning_rate, 
                           discount_factor=discount_factor,
                           start_exploration_prob=start_exploration_prob,
                           final_exploration_prob=final_exploration_prob,
                           exploration_decay=exploration_decay,
                           max_values=max_values,
                           min_values=min_values,
                           num_bins=num_bins)
    
    # Execute the Q-learning algorithm.
    rewards = q_learning( agent, 
                n_episodes=n_episodes)
    
    # Save the trained q_table and the parameters.
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "q_table.npy"), agent.q_table)
    
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump({
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "start_exploration_prob":start_exploration_prob,
                    "final_exploration_prob":final_exploration_prob,
                    "exploration_decay": exploration_decay,
                    "max_values": max_values,
                    "min_values": min_values,
                    "num_bins":num_bins,
                    "n_episodes":n_episodes,
                    "agent_dir":save_dir
            }, f)
        
    # Plot rewards
    agent.plot_rewards(rewards, os.path.join(save_dir, "rewards.png"))
    
    print("\n=== Script finished ===")
    print(f"   - It took {round((time.time()-t0)/60,2)} minutes to complete")
    print(f"   - Results on {save_dir}")
        
def parse_args(args):
    """
    Extract and return individual parameters from a given dictionary.
    
    Parameters:
        - params: Dictionary of parameters.
        
    Returns:
        Tuple of individual parameters.
    """
    
    if args.params_path:
        with open(args.params_path, "r") as f:
            params = json.load(f)
    
        return (params["learning_rate"], params["discount_factor"], 
                params["start_exploration_prob"], params["final_exploration_prob"], params["exploration_decay"],
                params["max_values"], params["min_values"], 
                params["num_bins"], int(params["n_episodes"]),
                params["save_dir"])
        
    else:
        
        return (args.learning_rate, args.discount_factor, 
            args.start_exploration_prob, args.final_exploration_prob, args.exploration_decay, 
            args.max_values, args.min_values, args.num_bins, 
            args.n_episodes, args.save_dir)

        
        
