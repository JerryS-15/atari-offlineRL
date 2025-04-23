import d3rlpy
import numpy as np
import argparse

def toMDP(args, chunk=int(1e5)):
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"


    actions = np.load(f"./buffers/{buffer_name}_action.npy")
    rewards = np.load(f"./buffers/{buffer_name}_reward.npy")
    terminals = np.load(f"./buffers/{buffer_name}_not_done.npy")

    crt_size = rewards.shape[0]
    print("Data size: ", crt_size)

    # observations = np.array

    crt = 0
    end = min(chunk, crt_size + 1)
    while crt < crt_size + 1:
        temp = np.load(f"./buffers/{buffer_name}_state_{end}.npy")
        if end == chunk:
            observations = temp
        else:
            observations = np.concatenate(observations, temp)
        crt = end
        end = min(end + chunk, crt_size + 1)

    dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    )

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

    data = toMDP(args)

    print(type(data))
    print(data.shape())