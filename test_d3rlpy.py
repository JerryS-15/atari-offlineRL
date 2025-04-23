import gym
import d3rlpy
from d3rlpy.logging import WanDBAdapterFactory
from pyvirtualdisplay import Display
import wandb
from d3rlpy.envs import AtariPreprocessing
import torch
from d3rlpy.models.encoders import PixelEncoderFactory

# wandb.init(project="d3rlpy", name="dqn-CartPole", monitor_gym=True)

def eval_callback(algo, epoch, total_step):
    episode_rewards = []
    for _ in range(5):  # 每轮评估 5 个 episode
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action = algo.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"[Eval] Epoch {epoch}, Average Reward: {avg_reward:.2f}")
    wandb.log({"eval/avg_reward": avg_reward, "epoch": epoch}, step=total_step)

# env = AtariPreprocessing(gym.make('PongNoFrameskip-v4'))
# eval_env = AtariPreprocessing(gym.make('PongNoFrameskip-v4'))
env = gym.make('PongNoFrameskip-v4')
eval_env = gym.make('PongNoFrameskip-v4')

# setup DQN algorithm
# dqn = d3rlpy.algos.DQNConfig(
#     learning_rate=1e-3,
#     target_update_interval=100,
# ).create(device='cpu')
dqn = d3rlpy.algos.DQNConfig(
    learning_rate=1e-4,  
    target_update_interval=1000,
    batch_size=32,
    encoder_factory=PixelEncoderFactory(),  # Use CNN
).create(device='cuda' if torch.cuda.is_available() else 'cpu')

# setup explorer
explorer = d3rlpy.algos.ConstantEpsilonGreedy(epsilon=0.3)
# setup replay buffer
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)

steps_per_epoch=50000

dqn.build_with_env(env)

wandb_factory = WanDBAdapterFactory(project="d3rlpy")

print(type(wandb_factory))
# wandb_logger = wandb_factory.create(algo=dqn, experiment_name="DQN-cartpole", n_steps_per_epoch=steps_per_epoch)


# start training
dqn.fit_online(
    env,
    buffer,
    explorer=d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.01,
        duration=1000000  # 100万步内线性衰减
    ),
    eval_env=eval_env,
    n_steps=5000000,
    n_steps_per_epoch=steps_per_epoch,
    logger_adapter=wandb_factory
)

from gym.wrappers import RecordVideo

# start virtual display
# d3rlpy.notebook_utils.start_virtual_display()

# wrap Monitor wrapper
env = RecordVideo(gym.make("PongNoFrameskip-v4", render_mode="rgb_array"), './video')

wrapped_env = AtariPreprocessing(env)

# evaluate
d3rlpy.metrics.evaluate_qlearning_with_environment(dqn, wrapped_env)

# d3rlpy.notebook_utils.render_video("video/rl-video-episode-0.mp4")
