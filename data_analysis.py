import numpy as np
import os

positive_threshold = 0
reward_file = "buffers/Default_PongNoFrameskip-v4_0_reward.npy"
rewards = np.load(reward_file)

positive_mask = rewards > positive_threshold
positive_count = np.sum(positive_mask)
total_samples = len(rewards)

print(f"rewards: {rewards}")
print("--------------------------------------")
print(f"Amount of Positive Data: {positive_count}/{total_samples} ({positive_count/total_samples:.2%})")
print("--------------------------------------")
