import d3rlpy
import numpy as np
import argparse
import os

def toMDP(args, chunk=int(1e5)):
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    print(f"[INFO] Loading buffer: {buffer_name}")
    actions = np.load(f"./buffers/{buffer_name}_action.npy")
    print(f"[INFO] Actions loaded. Shape: {actions.shape}")
    rewards = np.load(f"./buffers/{buffer_name}_reward.npy")
    print(f"[INFO] Rewards loaded. Shape: {rewards.shape}")
    terminals = np.load(f"./buffers/{buffer_name}_not_done.npy")
    terminals = 1 - terminals  # not_done -> done
    print(f"[INFO] Terminals computed. Shape: {terminals.shape}")

    crt_size = rewards.shape[0]
    # print("Data size: ", crt_size)

    # observations = np.array

    crt = 0
    end = min(chunk, crt_size + 1)
    k = 1
    while crt < crt_size + 1:
        temp = np.load(f"./buffers/{buffer_name}_state_{end}.npy")
        if end == chunk:
            observations = temp
            print(f"[INFO] Observation chunk {k} loaded, Shape: {observations.shape}.")
        else:
            observations = np.concatenate((observations, temp))
            print(f"[INFO] Observation chunk {k} loaded, Shape: {observations.shape}.")
        crt = end
        end = min(end + chunk, crt_size + 1)
        k = k + 1
    print(f"[INFO] Observation data fully loaded. Shape:", observations.shape, ", Type:", observations.dtype)

    print("[INFO] Before creating MDPDataset:")
    print(f"Observations Type: {type(observations)}, Shape: {observations.shape}")
    print(f"Actions Type: {type(actions)}, Shape: {actions.shape}")
    print(f"Rewards Type: {type(rewards)}, Shape: {rewards.shape}")
    print(f"Terminals Type: {type(terminals)}, Shape: {terminals.shape}")

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    print(f"[SUCCESS] Dataset created successfully!")

    print("[INFO] After creating MDPDataset:")
    print(f"Dataset Type: {type(dataset)}")
    print(f"Dataset Shape: {dataset.size()}")

    print("[INFO] Dataset Summary:")
    print(f"Episodes[0]: {dataset.episodes[0].size()}, dtype: {type(dataset.episodes[0])}")
    print(f"{dataset.episodes[0]}")
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
