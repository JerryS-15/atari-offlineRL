import numpy as np
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import d3rlpy
import os
import utils



def toMDP(args, chunk=int(1e5), stack=4):
    setting = f"{args.game}_{args.seed}"
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

    print("Preparing frame stacks ...")

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

    print("[INFO] Before creating MDPDataset:")
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
        device="cuda:5" if torch.cuda.is_available() else "cpu:5"
    )

    print(f"[SUCCESS] Dataset created successfully!")

    print("[INFO] After creating MDPDataset:")
    print(f"Dataset Type: {type(dataset)}")
    print(f"Dataset Shape: {dataset.size()}")

    return dataset



if __name__ == "__main__":

    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    parameters = {
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
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()

    print("------------------------------")
    # dataset = toMDP(args)
    with open(f"./d3Buffers/{args.game}_converted.h5", "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    print("Dataset Loaded.")
    print("------------------------------")
    ctx = "cuda:5" if torch.cuda.is_available() else "cpu:5"
    print(f"Using device ctx: {ctx}")

    print("Dataset structure: ")
    print("dataset.episodes[0]: ", dataset.episodes[0])

    gpu_dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array([observation for episode in dataset.episodes for observation in episode.observations]),
        actions=np.array([action for episode in dataset.episodes for action in episode.actions]),
        rewards=np.array([reward for episode in dataset.episodes for reward in episode.rewards]),
        terminals=np.array([terminal for episode in dataset.episodes for terminal in episode.terminated]),
        device=ctx  # Assign to device
    )

    eval_env, _, _, _ = utils.make_env(args.game, atari_preprocessing)

    batch_size = parameters["batch_size"]
    context_size = parameters["context_size"]
    target_return = parameters["target_return"]
    learning_rate = parameters["learning_rate"]

    max_timestep = 0
    for episode in dataset.episodes:
        max_timestep = max(max_timestep, episode.transition_count + 1)
    print(f"[INFO] Max timestep in dataset: {max_timestep}")

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
    ).create(device=ctx)

    num_epoch = 1  # 5 -> 1
    n_steps_per_epoch = dataset.transition_count // batch_size
    n_steps = n_steps_per_epoch * num_epoch
    print("[INFO] Starting training... please wait for epoch to begin.")

    dt.fit(
        gpu_dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        # eval_env=env,
        eval_target_return=target_return,
        eval_action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
            temperature=1.0,
        ),
        experiment_name=f"DiscreteDT_{args.game}_{args.seed}",
        # logger_adapter= wandb_factory,
    )

    if not os.path.exists("./models"):
        os.makedirs("./models")

    dt.save_policy(f"./models/d3rlpy_dt_policy_{args.game}_{args.seed}_test2.pt")

    actor = dt.as_stateful_wrapper(target_return)

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

