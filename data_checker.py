import numpy as np
import h5py

# data = np.load("./d3rlpy_data/cartpole_replay_v1.1.0.h5", allow_pickle=True)

# print(type(data))
# print(len(data))

with h5py.File('./d3rlpy_data/cartpole_replay_v1.1.0.h5', 'r') as f:
    print(f.keys())

    actions = f['actions'][:]
    discrete_action = f['discrete_action'][()]
    episode_terminals = f['episode_terminals'][:]
    observations = f['observations'][:]
    rewards = f['rewards'][:]
    terminals = f['terminals'][:]
    version = f['version'][()]

print("actions:")
print(type(actions))
print(len(actions))

print("discrete_actions:")
print(type(discrete_action))
print(discrete_action)

print("episode_terminals:")
print(type(episode_terminals))
print(len(episode_terminals))