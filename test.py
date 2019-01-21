from unityagents import UnityEnvironment
import numpy as np
import agent
import argparse
import torch
from collections import deque
import matplotlib.pyplot as plt


parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--total_episodes',default=600,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--env',default='Banana_Linux_NoVis/Banana.x86',type=str,required=False,help='(default=%(default)d)')
opt=parser.parse_args()


env = UnityEnvironment(file_name=opt.env)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
Agent = agent.Agent
agent_ = Agent(state_size=state_size, action_size=action_size, seed=0, total_episodes=opt.total_episodes)


agent_.qnetwork_local.load_state_dict(torch.load('checkpoint_banana.pth'))
env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
while True:
    action = agent_.act(state)  # select an action
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    if done:  # exit loop if episode finished
        break

print("Score: {}".format(score))