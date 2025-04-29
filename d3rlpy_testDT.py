import argparse
import numpy as np
import torch
import wandb
from d3rlpy.logging import WanDBAdapterFactory
import os

import d3rlpy
import gym

# debug
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)


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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pre-stack", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    print("------------------------------")
    dataset = toMDP(args)
    print("------------------------------")

    env = gym.make(args.game)

    # dataset, env = d3rlpy.datasets.get_atari_transitions(
    #     args.game,
    #     fraction=0.01,
    #     index=1 if args.game == "asterix" else 0,
    #     num_stack=4,
    #     sticky_action=False,
    #     pre_stack=args.pre_stack,
    # )

    d3rlpy.envs.seed_env(env, args.seed)

    if args.game == "PongNoFrameskip-v4":
        batch_size = 256  # 512
        context_size = 50  # 50
    else:
        batch_size = 128
        context_size = 30

    if args.game == "PongNoFrameskip-v4":
        target_return = 20
    elif args.game == "breakout":
        target_return = 90
    elif args.game == "qbert":
        target_return = 2500
    elif args.game == "seaquest":
        target_return = 1450
    else:
        raise ValueError(f"target_return is not defined for {args.game}")
    
    wandb.init(
        project="d3rlpy",
        name=f"DiscreteDT_{args.game}_{args.seed}",
        config={
            "game": args.game,
            "seed": args.seed,
            "context_size": context_size,
            "batch_size": batch_size,
            "target_return": target_return,
            # "learning_rate": 6e-4,
            "algo": "DiscreteDecisionTransformer"
        }
    )

    wandb_factory = WanDBAdapterFactory(project="d3rlpy")

    # extract maximum timestep in dataset
    max_timestep = 0
    for episode in dataset.episodes:
        max_timestep = max(max_timestep, episode.transition_count + 1)
    print(f"[INFO] Max timestep in dataset: {max_timestep}")

    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(
        batch_size=batch_size,
        context_size=context_size,
        learning_rate=6e-4,
        activation_type="gelu",
        embed_activation_type="tanh",
        encoder_factory=d3rlpy.models.PixelEncoderFactory(
            feature_size=128, exclude_last_activation=True
        ),  # Nature DQN
        num_heads=8,
        num_layers=6,
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
    ).create(device=args.gpu)

    n_steps_per_epoch = dataset.transition_count // batch_size
    n_steps = n_steps_per_epoch * 5
    print("[INFO] Starting training... please wait for epoch to begin.")
    dt.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        eval_env=env,
        eval_target_return=target_return,
        eval_action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
            temperature=1.0,
        ),
        experiment_name=f"DiscreteDT_{args.game}_{args.seed}",
        logger_adapter= wandb_factory,
    )


if __name__ == "__main__":
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    main()
    wandb.finish()