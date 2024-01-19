import numpy as np
import pickle
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Any

class QLearningAgent:
    def __init__(self, env: Any, learning_rate: float = 0.1, discount_factor: float = 0.99,
                start_exploration_prob: float = 1.0, final_exploration_prob: float = 0.01, 
                exploration_decay: float = 1e4, max_values: List[float] = [2.4, 3.0, 0.3, 3.0], 
                min_values: List[float] = [-2.4, -3.0, -0.3, -3.0], num_bins: List[int] = [20, 20, 20, 20]) -> None:
        """
        Initialize the Q-Learning Agent.
        
        Parameters:
        - env: The environment the agent will operate in.
        - learning_rate: The rate at which the agent learns from new experiences (default is 0.1).
        - discount_factor: The factor to discount future rewards (default is 0.99).
        - start_exploration_prob: The initial probability for the agent to explore rather than exploit (default is 1.0).
        - final_exploration_prob: The final probability for the agent to explore (default is 0.01).
        - exploration_decay: The number of steps for the exploration rate to reach final_exploration_prob (default is 1e4).
        - max_values: The maximum values for the state variables in the environment (default is [2.4, 3.0, 0.3, 3.0]).
        - min_values: The minimum values for the state variables in the environment (default is [-2.4, -3.0, -0.3, -3.0]).
        - num_bins: Number of bins to discretize the state variables into (default is [20, 20, 20, 20]).
        
        Attributes:
        - self.env: The environment.
        - self.n_actions: Number of possible actions.
        - self.learning_rate: Learning rate.
        - self.discount_factor: Discount factor.
        - self.exploration_prob: Current exploration probability.
        - self.start_exploration_prob: Starting exploration probability.
        - self.final_exploration_prob: Final exploration probability.
        - self.exploration_decay: Exploration decay rate.
        - self.q_table: The Q-table for storing state-action values, initialized to zeros.
        - self.max_values: Maximum state variables.
        - self.min_values: Minimum state variables.
        - self.num_bins: Number of bins for each state variable.
        """
        
        self.env = env
        self.n_actions = self.env.action_space.n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = start_exploration_prob
        self.start_exploration_prob = start_exploration_prob
        self.final_exploration_prob = final_exploration_prob
        self.exploration_decay = int(exploration_decay)
        self.m = (self.final_exploration_prob - self.start_exploration_prob) / self.exploration_decay
        self.q_table = np.zeros(tuple(num_bins) + (self.n_actions,))
        self.max_values = max_values
        self.min_values = min_values
        self.num_bins = num_bins

    def choose_action(self, state: Tuple[int, int, int, int]) -> int:
        """
        Choose an action based on the current state.
        
        With a probability of self.exploration_prob, the agent chooses a random action (exploration).
        Otherwise, it chooses the action with the highest Q-value for the current state (exploitation).
        
        Parameters:
        - state: The current state of the environment.
        
        Returns:
        - int: The action chosen by the agent.
        """
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def act(self, state: Tuple[int, int, int, int]) -> int:
        """
        Choose the best action for the given state based on the Q-table.
        
        Parameters:
        - state: The current state of the environment.
        
        Returns:
        - int: The action with the highest Q-value for the given state.
        """
        return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple[int, int, int, int], action: int, reward: float, next_state: Tuple[int, int, int, int]) -> None:
        """
        Update the Q-value for a given state-action pair based on the reward received and the next state.
        
        The Q-value is updated according to the formula:
        Q(s, a) = (1 - learning_rate) * Q(s, a) + learning_rate * (reward + discount_factor * max(Q(s', a')))
        
        Parameters:
        - state: The current state of the environment.
        - action: The action taken by the agent.
        - reward: The reward received after taking the action.
        - next_state: The state transitioned into after taking the action.
        """
        # Retrieve the old Q-value from the Q-table using the current state and action.
        # This is the Q-value prior to the update.
        old_q_value = self.q_table[state][action]

        # Calculate the maximum Q-value for the next state.
        # This is used in the Q-value update equation to incorporate future rewards.
        next_q_value_max = np.max(self.q_table[next_state])

        # Update the Q-value for the current state-action pair based on the update equation.
        # 1. (1 - self.learning_rate) * old_q_value: This part of the equation is responsible for
        #    keeping some proportion of the old Q-value.
        # 2. self.learning_rate * (reward + self.discount_factor * next_q_value_max): This part of the
        #    equation is responsible for updating the Q-value with the new information. It uses the
        #    immediate reward and the discounted maximum future Q-value (for the next state).
        new_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * next_q_value_max)

        # Store the new Q-value back into the Q-table to replace the old Q-value.
        self.q_table[state][action] = new_value

    def decay_exploration(self, step: int) -> None:
        """
        Update the exploration probability based on the current step.
        
        The exploration probability decays linearly from start_exploration_prob to final_exploration_prob
        over a number of steps specified by exploration_decay. 
        
        Parameters:
        - step: The current step (episode or iteration) in the training process.
        """
        
        # Update the exploration probability value
        self.exploration_prob = self.start_exploration_prob + self.m * step

        # Ensure exploration probability does not go below final_exploration_prob
        self.exploration_prob = max(self.final_exploration_prob, self.exploration_prob)

    def discretize_state(self, state: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """
        Convert a continuous state to a discrete state.
        
        The state space is divided into bins specified by num_bins for each dimension.
        The function then determines which bin each state variable belongs to.
        
        Parameters:
        - state: The current continuous state of the environment.
        
        Returns:
        - tuple: The discretized state.
        """
        # Calculate the size of each bin for the 4 dimensions of the state space.
        # The size is determined by the range (max - min) of each dimension divided by the number of bins for that dimension.
        bin_size = [(self.max_values[i] - self.min_values[i]) / self.num_bins[i] for i in range(4)]

        # Initialize an empty list to hold the discretized state variables.
        discrete_state = []

        # Loop through each dimension in the state space (4 dimensions in this case).
        for i in range(4):

            # Calculate which bin the current state variable falls into.
            # This is done by subtracting the minimum value for that dimension from the state variable,
            # then dividing by the size of the bin for that dimension.
            discrete_value = int((state[i] - self.min_values[i]) / bin_size[i])

            # Ensure the bin index (discrete_value) is within the valid range [0, num_bins[i] - 1].
            # If the calculated bin index is below 0, it's set to 0.
            # If the calculated bin index is above (num_bins[i] - 1), it's set to (num_bins[i] - 1).
            discrete_value = min(self.num_bins[i] - 1, max(0, discrete_value))

            # Add the calculated discrete_value to the list of discretized state variables.
            discrete_state.append(discrete_value)
        
        # Convert the list of discretized state variables to a tuple and return it.
        return tuple(discrete_state)
    
    def save(self, file_name: str) -> None:
        """
        Save the current state of the agent to a file using pickle.
        
        Parameters:
        - file_name: The name of the file to save the agent's state.
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            
    def play(self, n_episodes: int = 10, save_gif_path: Optional[str] = None) -> Tuple[List[float], np.ndarray]:
        """
        Run the agent through a series of episodes in the environment without learning.

        Parameters:
        - n_episodes: Number of episodes to run (default is 10).
        - save_gif_path: Optional path to save a GIF of the gameplay (default is None).

        Returns:
        - rewards: List of rewards received per episode.
        - np.array(frames): Numpy array containing the frames for each step in all episodes.
        """
        
        # Initialize empty lists to hold rewards and frames.
        rewards, frames = [], []

        # Loop over each episode.
        with tqdm(total=n_episodes, desc="Playing Cartpole") as pbar:
            for episode in range(n_episodes):
                # Reset the environment and get the initial state.
                state, _ = self.env.reset()
                
                # Discretize the initial state.
                state = self.discretize_state(state)

                # Render the environment and store the frame.
                frames.append(self.env.render())

                # Initialize a variable to hold the total reward for the episode.
                reward_episode = 0

                # Start the episode loop.
                while True:

                    # Choose an action based on the current state.
                    action = self.act(state)

                    # Take the action and observe the next state, reward, and whether the episode has terminated.
                    next_state, reward, terminated, truncated, _ = self.env.step(action)

                    # Discretize the next state.
                    next_state = self.discretize_state(next_state)

                    # Render the environment and store the frame.
                    frames.append(self.env.render())

                    # Check if the episode has terminated or is truncated.
                    if terminated or truncated:
                        break
                    else:
                        # Accumulate the reward for the current episode.
                        reward_episode += reward

                        # Update the current state to the next state.
                        state = next_state

                # Store the total reward for the episode in the rewards list.
                rewards.append(reward_episode)
                
                # Update progress bar
                pbar.set_postfix(reward=reward_episode)
                pbar.update(1)

            # If a path is provided, save the frames as a GIF.
            if save_gif_path is not None:
                self.make_gif(frames, save_gif_path)

            # Return the list of rewards and the frames as a NumPy array.
        return rewards, np.array(frames)

    def make_gif(self, frames: List[Any], save_path: str, duration: float = 1/40) -> None:
        """
        Create a GIF from a list of frames.

        Parameters:
        - frames: List of frames to include in the GIF.
        - save_path: The path where the GIF will be saved.
        - duration: The duration for each frame in the GIF (default is 1/40).
        """
        imageio.mimsave(save_path, frames, format="GIF", duration=duration)
    
    def plot_rewards(self, rewards: List[float], save_path: Optional[str] = None) -> None:
        """
        Plot rewards over episodes.
        
        Parameters:
        - rewards (list): List of rewards obtained over episodes.
        - save_path (str, optional): Path where the plot will be saved. If None, the plot is not saved.
        """
        
        # Set Seaborn theme style
        sns.set_theme(style="darkgrid")
        
        # Create the plot using subplots
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rewards, color="dodgerblue", lw=2)
        ax.set_title("Rewards through episodes", fontsize=16)
        ax.set_xlabel("Episodes", fontsize=14)
        ax.set_ylabel("Reward", fontsize=14)
        
        # If save_path is provided, save the plot to the specified path
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, format='png')
        
        plt.close() 
