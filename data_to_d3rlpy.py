import d3rlpy
import numpy as np
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def toMDP(args, chunk=int(1e5), stack=4):
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    buffer_dir = "./buffers"

    print(f"[INFO] Loading buffer: {buffer_name}")

    # Action, reward, terminals
    actions = np.load(f"./buffers/{buffer_name}_action.npy")
    print(f"[INFO] Actions loaded. Shape: {actions.shape}")
    rewards = np.load(f"./buffers/{buffer_name}_reward.npy")
    print(f"[INFO] Rewards loaded. Shape: {rewards.shape}")
    terminals = np.load(f"./buffers/{buffer_name}_not_done.npy")
    terminals = 1 - terminals  # not_done -> done
    print(f"[INFO] Terminals computed. Shape: {terminals.shape}")

    # crt_size = rewards.shape[0]
    # print("Data size: ", crt_size)

    print(f"[INFO] Observation loading starts.")

    total_samples = rewards.shape[0]
    num_chunks = (total_samples + chunk - 1) // chunk
    
    # Pre-allocate space
    observations = np.empty((total_samples, 84, 84), dtype=np.uint8)
    
    # Concurrently loading all chunks
    def load_chunk(chunk_idx):
        start = chunk_idx * chunk
        end = min((chunk_idx + 1) * chunk, total_samples)
        path = os.path.join(buffer_dir, f"{buffer_name}_state_{end}.npy")
        return chunk_idx, np.load(path)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for chunk_idx, chunk_data in tqdm(
            executor.map(load_chunk, range(num_chunks)),
            total=num_chunks,
            desc="Progress"
        ):
            start = chunk_idx * chunk
            end = min((chunk_idx + 1) * chunk, total_samples)
            observations[start:end] = chunk_data
    
    print(f"[INFO] Observation data fully loaded. Shape:", observations.shape, ", Type:", observations.dtype)

    # observations = np.array

    # crt = 0
    # end = min(chunk, crt_size + 1)
    # k = 1
    # while crt < crt_size + 1:
    #     temp = np.load(f"./buffers/{buffer_name}_state_{end}.npy")
    #     if end == chunk:
    #         observations = temp
    #         print(f"[INFO] Observation chunk {k} loaded, Shape: {observations.shape}.")
    #     else:
    #         observations = np.concatenate((observations, temp))
    #         print(f"[INFO] Observation chunk {k} loaded, Shape: {observations.shape}.")
    #     crt = end
    #     end = min(end + chunk, crt_size + 1)
    #     k = k + 1
    # print(f"[INFO] Observation data fully loaded. Shape:", observations.shape, ", Type:", observations.dtype)

    print("Preparing frame stacks ...")

    # obs = []
    # # next_obs = []
    # valid_idx = []

    # for i in range(stack - 1, crt_size):
    #     # Check if any early termination happened within the stack (optional)
    #     is_valid = True
    #     for j in range(i - stack + 1, i):
    #         if terminals[j] == 1:  # episode ended
    #             print("... A whole episode has finished frame stacking.")
    #             is_valid = False
    #             break
    #     if not is_valid:
    #         continue

    #     obs.append(observations[i - stack + 1:i + 1])         # shape: (4, 84, 84)
    #     # next_obs.append(observations[i - stack + 2:i + 2])    # shape: (4, 84, 84)
    #     valid_idx.append(i)

    # obs = np.stack(obs)  # (N, 4, 84, 84)
    # # next_obs = np.stack(next_obs)
    # valid_idx = np.array(valid_idx)

    done_indices = np.where(terminals == 1)[0]
    
    # Get valid mask (invalid from step stack_frames-1)
    valid_mask = np.ones(total_samples, dtype=bool)
    for idx in done_indices:
        valid_mask[idx+1 : idx+stack] = False
    
    # Compute range of valid indices
    valid_indices = np.arange(stack-1, total_samples)[valid_mask[stack-1:]]
    
    # as_strided used for stacking!
    def stack_frames_zero_copy(arr, stack):
        shape = (len(arr) - stack + 1, stack, *arr.shape[1:])
        strides = (arr.strides[0], *arr.strides)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    
    stacked_obs = stack_frames_zero_copy(observations, stack)
    stacked_obs = stacked_obs[valid_indices - (stack-1)]

    print(f"[INFO] Final stacked observations: {stacked_obs.shape}")

    actions = actions[valid_indices]
    rewards = rewards[valid_indices]
    terminals = terminals[valid_indices]

    print("[INFO] Frame stacking finished.")

    # Add channel dimension for grayscale image (1, 84, 84)
    # observations = observations[:, np.newaxis, :, :]
    # observations = np.repeat(observations[:, np.newaxis, :, :], 4, axis=1)

    print("[INFO] Before creating MDPDataset:")
    # print(f"Observations Type: {type(obs)}, Shape: {obs.shape}")
    print(f"Observations Type: {type(stacked_obs)}, Shape: {stacked_obs.shape}")
    print(f"Actions Type: {type(actions)}, Shape: {actions.shape}")
    print(f"Rewards Type: {type(rewards)}, Shape: {rewards.shape}")
    print(f"Terminals Type: {type(terminals)}, Shape: {terminals.shape}")

    print(f"[INFO] Creating MDPDataset...")

    dataset = d3rlpy.dataset.MDPDataset(
        observations=stacked_obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        # next_observations=next_obs
    )

    print(f"[SUCCESS] Dataset created successfully!")

    print("[INFO] After creating MDPDataset:")
    print(f"Dataset Type: {type(dataset)}")
    print(f"Dataset Shape: {dataset.size()}")

    # print("[INFO] Dataset Summary:")
    # print(f"Episodes[0]: {dataset.episodes[0].size()}, dtype: {type(dataset.episodes[0])}")
    # print(f"{dataset.episodes[0]}")
    # print(f"Observation shape: {dataset.episodes[0].observation.size()}, dtype: {type(dataset.episodes[0].observation)}")
    # print(f"Action shape: {dataset.actions.shape}, dtype: {dataset.actions.dtype}")
    # print(f"Reward shape: {dataset.rewards.shape}, dtype: {dataset.rewards.dtype}")
    # print(f"Terminal shape: {dataset.terminals.shape}, dtype: {dataset.terminals.dtype}")

    # Get sample content for reference
    # print("\n[INFO] Sample transitions:")
    # for i in range(3):
    #     obs, act, rew, next_obs, done = dataset[i]
    #     print(f"Sample {i}:")
    #     print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    #     print(f"  action    : {act}, type: {type(act)}")
    #     print(f"  reward    : {rew}, type: {type(rew)}")
    #     print(f"  next_obs shape: {next_obs.shape}, dtype: {next_obs.dtype}")
    #     print(f"  done      : {done}, type: {type(done)}")


    return dataset



if __name__ == "__main__":

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PongNoFrameskip-v4")     # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.2, type=float)# Threshold hyper-parameter for BCQ   <- Original default is 0.3 -> tried 0.2 with depth 1024
    parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--train_cql", action="store_true", help="Train with Conservative Q-Learning")
    parser.add_argument("--CQL_alpha", default=1.0, type=float, help="Regularization strength for CQL")

    args = parser.parse_args()

    print("------------------------------")

    data = toMDP(args)

    print("All data fully loaded.")
    print("------------------------------")

    if not os.path.exists("./d3Buffers"):
        os.makedirs("./d3Buffers")

    print("Generating .h5 file.")

    # data.dump(f"./d3Buffers/{args.env}_converted.h5")
    with open(f"./d3Buffers/{args.env}_converted.h5", "w+b") as f:
        data.dump(f)

    print("Converted file generated. All done!")

    # with open(f"./d3Buffers/{args.env}_converted.h5", "rb") as f:
    #     parsed_dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

    # print(type(parsed_dataset))

    # print(type(data))
    # print(data.shape())
