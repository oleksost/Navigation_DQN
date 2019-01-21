from unityagents import UnityEnvironment
import numpy as np
import agent
import argparse
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

def dqn(n_episodes=opt.total_episodes, max_t=1000, eps_start=1.0, eps_end=0.001, eps_decay=0.95):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent_.act(state, eps)
            # next_state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]
            reward, next_state, done = env_info.rewards[0], env_info.vector_observations[0], env_info.local_done[0]
            agent_.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return scores

scores = dqn()
torch.save(agent_.qnetwork_local.cpu().state_dict(), 'checkpoint_banana.pth')

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('Rewards.pdf', format='pdf', dpi=300)

