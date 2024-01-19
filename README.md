# **QLearning CartPole: A python library for training QLearning agents to play CartPole**

This repository contains a Python package for training an artificial agent to play the CartPole game using the Q-Learning algorithm.

### 1. Features

- Q-Learning algorithm implemented for CartPole
- Command-line tools for training and playing
- Easy and flexible configuration

### 2. Requirements

- Python 3.12 or higher
- Python packages in requirements.txt
- Python pip

### 3. Installation

To install this package, user needs to activate a python environment with python >=3.12. For example:

`conda create -n cartpole python==3.12`
`conda activate cartpole`

Then, just run the following command:

`pip install -r requirements.txt`
`pip install .`

### 4. Usage

A part from the classes and functions defined in the package that user can import and wrap around as wished, the package provides two command-line tools: `cartpole_train` and `cartpole_play`.

#### Training

To train a new agent, user can simply run:

`cartpole_train --save_dir /especify_your_desired_dir`

This will train it and store the results (the Q-Table and the parameters used) in the specified directory. Notice that this command will run the training with the default values of the parameters. In order to see all the tuneable parameters, user can just run `cartpole_train -h` (or check the main function in `cartpole/learn.py`).

By default, `cartpole_train` tool will use the parameters in `config/cartpole_train.json`.

#### Playing

In order to play with a trained agent, user can simply run:

`cartpole_play --agent_dir /dir_where_the_training_took_place --save_dir /dir_where_to_save_playing_results`

Again, there are other parameters user can check by running `cartpole_play -h` (or checking the main function in `cartpole/play.py`).

### 5. Contribution

Feel free to open an issue or pull request to improve this package.

### 6. License

This project is under the MIT license.
