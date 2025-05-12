import argparse
import numpy as np
import torch
import wandb
from d3rlpy.logging import WanDBAdapterFactory
# from d3rlpy.envs import wrap_env
import os

import d3rlpy
import gym
import utils

# debug
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# torch.set_num_threads(1)


def toMDP(args, chunk=int(1e5)):
    setting = f"{args.game}_{args.seed}"
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
            print(f"[INFO] Observation chunk {k} loaded")
        else:
            observations = np.concatenate((observations, temp))
            print(f"[INFO] Observation chunk {k} loaded.")
        crt = end
        end = min(end + chunk, crt_size + 1)
        k = k + 1
    print("[INFO] Observation data fully loaded.")

    # Add channel dimension for grayscale image (1, 84, 84)
    observations = observations[:, np.newaxis, :, :]

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    print(f"[SUCCESS] Dataset created successfully!")
    return dataset

def eval_policy(policy, env_name, seed, eval_episodes=10, target_return=20, temperature=1.0):
    eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
    eval_env.seed(seed + 100)
    
    # action_sampler = d3rlpy.algos.SoftmaxTransformerActionSampler(temperature=temperature)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state = eval_env.reset()
        done = False

        # Initialize trajectory
        states = []
        actions = []
        rewards = []
        timesteps = []

        rtg = target_return # Initialize return-to-go
        episode_reward = 0.
        t = 0

        while not done:

            timesteps.append(t)
            t += 1

            # Action prediction
            action = policy(
                states=states,
                actions=actions,
                rewards=rewards,
                target_return=rtg,
                timesteps=timesteps,
                temperature=temperature
            )

            next_state, reward, done, _ = eval_env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            rtg -= reward
            episode_reward += reward

        avg_reward += episode_reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    wandb.log({"Evaluation Reward": avg_reward, "Evaluation Episodes": eval_episodes})

    return avg_reward

def main(args, parameters) -> None:

    print("------------------------------")
    # dataset = toMDP(args)
    with open(f"./d3Buffers/{args.game}_converted.h5", "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    print("Dataset Loaded.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("------------------------------")

    # env = gym.make(args.game)
    eval_env, _, _, _ = utils.make_env(args.game, atari_preprocessing)
    # env.seed(args.seed)

    # print(type(eval_env))
    # print(eval_env.observation_space.shape)

    # dataset, env = d3rlpy.datasets.get_atari_transitions(
    #     args.game,
    #     fraction=0.01,
    #     index=1 if args.game == "asterix" else 0,
    #     num_stack=4,
    #     sticky_action=False,
    #     pre_stack=args.pre_stack,
    # )

    # d3rlpy.envs.seed_env(env, args.seed)

    batch_size = parameters["batch_size"]
    context_size = parameters["context_size"]
    target_return = parameters["target_return"]
    learning_rate = parameters["learning_rate"]

    # if args.game == "PongNoFrameskip-v4":
    #     batch_size = 256  # 512
    #     context_size = 5  # 50
    # else:
    #     batch_size = 128
    #     context_size = 30

    # if args.game == "PongNoFrameskip-v4":
    #     target_return = 20
    # elif args.game == "breakout":
    #     target_return = 90
    # elif args.game == "qbert":
    #     target_return = 2500
    # elif args.game == "seaquest":
    #     target_return = 1450
    # else:
    #     raise ValueError(f"target_return is not defined for {args.game}")

    wandb_factory = WanDBAdapterFactory(project="d3rlpy")

    # extract maximum timestep in dataset
    max_timestep = 0
    for episode in dataset.episodes:
        max_timestep = max(max_timestep, episode.transition_count + 1)
    print(f"[INFO] Max timestep in dataset: {max_timestep}")

    wandb.log({
        "max_timestep": max_timestep,
        "dataset_size": dataset.transition_count,
        "n_episodes": len(dataset.episodes),
    })

    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(
        batch_size=batch_size,
        context_size=context_size,
        learning_rate=learning_rate,
        activation_type="gelu",  # gelu
        embed_activation_type="tanh",
        encoder_factory=d3rlpy.models.PixelEncoderFactory(
            feature_size=128, exclude_last_activation=True
        ),  # Nature DQN
        num_heads=2,  # 8 -> 2
        num_layers=2,  # 6 -> 2
        attn_dropout=0.1,
        embed_dropout=0.1,
        optim_factory=d3rlpy.optimizers.GPTAdamWFactory(
            betas=(0.9, 0.95),
            weight_decay=0.1,
            clip_grad_norm=1.0,
        ),
        warmup_tokens=512 * 20,
        final_tokens=2 * 500000 * context_size * 3,
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        # observation_scaler=None,  # debug
        max_timestep=max_timestep,
        position_encoding_type=d3rlpy.PositionEncodingType.GLOBAL,
        compile_graph=args.compile,
    ).create(device='cuda' if torch.cuda.is_available() else 'cpu')

    num_epoch = 1  # 5 -> 1
    n_steps_per_epoch = dataset.transition_count // batch_size
    n_steps = n_steps_per_epoch * num_epoch
    print("[INFO] Starting training... please wait for epoch to begin.")

    # Check if image and model both on GPU
    obs = dataset.episodes[0].observations[0]
    print("Observation type:", type(obs))
    if isinstance(obs, torch.Tensor):
        print("Observation device:", obs.device)

    dt.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        # eval_env=env,
        eval_target_return=target_return,
        eval_action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
            temperature=1.0,
        ),
        experiment_name=f"DiscreteDT_{args.game}_{args.seed}",
        logger_adapter= wandb_factory,
    )

    if not os.path.exists("./models"):
        os.makedirs("./models")

    dt.save_policy(f"./models/d3rlpy_dt_policy_{args.game}_{args.seed}_test.pt") 

    # eval_policy(dt, args.game, args.seed, eval_episodes=10, target_return=target_return)

    # Interaction test for evaluation
    actor = dt.as_stateful_wrapper(target_return)
    # eval_env.seed(args.seed + 100)

    avg_reward = 0.
    eval_episodes = 10
    for _ in range(eval_episodes):
        state = eval_env.reset()
        done = False

        # Initialize trajectory
        states = []
        actions = []
        rewards = []
        timesteps = []

        rtg = target_return # Initialize return-to-go
        episode_reward = 0.0
        t = 0

        while not done:

            timesteps.append(t)
            t += 1

            # print("----------")
            # print(state.shape)
            # print(state)
            # print("----------")

            # Action prediction
            action = actor.predict(state, episode_reward)

            next_state, reward, done, _ = eval_env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            rtg -= reward
            episode_reward += reward
        
        actor.reset()

        avg_reward += episode_reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    wandb.log({"Evaluation Reward": avg_reward, "Evaluation Episodes": eval_episodes})


if __name__ == "__main__":

    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    atari_parameters = {
        "batch_size": 256,  # 512
        "context_size": 5,  # 50
        "target_return": 20,
        "learning_rate": 6e-4
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pre-stack", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    wandb.init(
        project="d3rlpy",
        name=f"DiscreteDT_{args.game}_{args.seed}",
        config=atari_parameters
    )

    print("---------------------------------------")
    if args.eval:
        print("Evaluation Mode.")
        model_path = f"./models/d3rlpy_dt_model_{args.game}_{args.seed}_test.pt"
        # policy = torch.jit.load(f"./models/d3rlpy_dt_model_{args.game}_{args.seed}.pt")
        dt = d3rlpy.algos.DiscreteDecisionTransformer.load_model(model_path)
        eval_policy(dt, args.game, args.seed, eval_episodes=10)
    else:
        print("Training Mode.")
        main(args, atari_parameters)
    print("---------------------------------------")
    wandb.finish()