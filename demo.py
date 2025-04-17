import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
import discrete_CQL
import discrete_BCQ
import DQN
import argparse
import utils

# MODEL_PATH = "./models/behavioral_{setting}"

def preprocess(obs):
    if obs is None:
        raise ValueError("obs is None")
    if not isinstance(obs, np.ndarray):
        raise ValueError("obs is not a valid image. Got: {}".format(type(obs)))
    # if len(obs.shape) != 3:
    #     raise ValueError(f"obs should have shape (H,W,3), but got {obs.shape}")
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    elif len(obs.shape) == 2:
        image = obs
    else:
        raise ValueError(f"Unsupported obs shape: {obs.shape}")
    image = cv2.resize(image, (84, 84))
    return image.astype(np.uint8)

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = []

    def reset(self, obs):
        self.frames = [obs for _ in range(self.k)]
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.pop(0)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0)

def render(obs, delay=0.01):
    # obs = np.transpose(obs, (1, 2, 0))[-1]  # 取最后一帧显示
    frame = obs[-1]
    cv2.imshow("Trained Pong", frame)
    cv2.waitKey(1)
    time.sleep(delay)

def run_demo(env, is_atari, num_actions, state_dim, device, args, parameters):
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    # policy = DQN.DQN(
    #     is_atari,
    #     num_actions,
    #     state_dim,
    #     device,
    #     parameters["discount"],
    #     parameters["optimizer"],
    #     parameters["optimizer_parameters"],
    #     parameters["polyak_target_update"],
    #     parameters["target_update_freq"],
    #     parameters["tau"],
    #     parameters["initial_eps"],
    #     parameters["end_eps"],
    #     parameters["eps_decay_period"],
    #     parameters["eval_eps"],
    # )
    policy = discrete_BCQ.discrete_BCQ(
		is_atari,
		num_actions,
		state_dim,
		device,
		args.BCQ_threshold,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"]
	)

    try:
        policy.load(f"./models/bcq_{setting}")
        print(f"✅ Model Loading Successfully from path.")
    except Exception as e:
        print(f"❌ Model Loading Failed: {str(e)}")
        return

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env = gym.make("PongNoFrameskip-v4")

    # model = DQN.DQN(frames=4, num_actions=env.action_space.n).to(device)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # model.eval()

    # # Random input to test the validity of model
    # dummy_input = torch.randn(1, 4, 84, 84).to(device)
    # output = model(dummy_input)

    # print("✅ Model Loading Successful, with output shape: ", output.shape)

    # obs = env.reset()
    # if isinstance(obs, tuple):
    #     obs = obs[0]

    # frame_stack = FrameStack(k=4)
    obs = env.reset()
    # obs = obs[0]
    # obs = preprocess(obs)
    # state = frame_stack.reset(obs)

    done = False
    total_reward = 0
    while not done:
        render(obs)
        # state_tensor = torch.FloatTensor(state / 255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            # action = model(state_tensor).argmax(1).item()
            action = policy.select_action(np.array(obs), eval=True)
        next_obs, reward, done, _ = env.step(action)
        # done = terminated or truncated
        total_reward += reward
        obs = next_obs
        # next_obs = next_obs[0]
        # next_obs = preprocess(next_obs)
        # state = frame_stack.step(next_obs)

    print("Game over. Total reward:", total_reward)
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Atari Specific
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    atari_parameters = {
        # Exploration
        "start_timesteps": 2e4,   # 2e4
        "initial_eps": 1,
        "end_eps": 1e-2,   # 1e-2
        "eps_decay_period": 1e5,   # 25e4 -> 1e5
        # Evaluation
        "eval_freq": 5e4,
        "eval_eps": 1e-3,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 128,   # 32 -> 64 -> 128
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 1e-4,   # 0.0000625
            "eps": 0.00015
        },
        "train_freq": 4,
        "polyak_target_update": False,
        "target_update_freq": 1000,    # 8e3 -> 1000
        "tau": 1
    }

    regular_parameters = {
        # Exploration
        "start_timesteps": 1e3,
        "initial_eps": 0.1,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        "eval_freq": 5e3,
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 64,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4
        },
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005
    }

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

    env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
    parameters = atari_parameters if is_atari else regular_parameters

    # Set seeds
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_demo(env, is_atari, num_actions, state_dim, device, args, parameters)