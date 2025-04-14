import csv
import matplotlib.pyplot as plt
import numpy as np

file_path = "./buffer_data/wandb_0407_dqn_128.csv"
threshold = 18

total_rows = 0
over_threshold = 0

min_reward = -21
max_reward = 21

rewards = []

with open(file_path, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)
    # print("Headers:", headers)
    for row in reader:
        total_rows += 1
        value = float(row[1])
        # print(value)
        # rewards.append(value)
        if value > threshold:
            over_threshold += 1
        try:
            reward = int(float(row[1]))
            if min_reward <= reward <= max_reward:
                rewards.append(reward)
        except (ValueError, IndexError):
            print("One Error line exists: Value Error ", ValueError, " Index Error ", IndexError)
            continue

print(f"Total number of episodes with rewards > {threshold} is: {over_threshold}/{total_rows} ({over_threshold/total_rows:.2%})")

rewards_array = np.array(rewards)

plt.close('all')
fig, ax = plt.subplots(figsize=(12, 6))

bins = np.arange(min_reward, max_reward + 2) - 0.5
ax.hist(rewards_array, bins=bins, edgecolor='black', align='mid')

ax.set_xlabel('Reward')
ax.set_ylabel('Counts')
ax.set_title(f'Reward Distribution (n={len(rewards_array)})')
ax.set_xticks(np.arange(min_reward, max_reward + 1))
ax.grid(axis='y')

plt.tight_layout()
plt.show()
