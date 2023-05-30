import hydra
import ray
import ray.rllib.algorithms.dqn as dqn
import ray.rllib.algorithms.ppo as ppo
import numpy as np
from ray.tune.registry import register_env
import tqdm

import memsearch
import memsearch
from memsearch.igridson_env import SMGFixedEnv
from memsearch.igridson_utils import make_scene_sampler_and_evolver_


from gym.core import ObservationWrapper
import gym.spaces as spaces
# class FlattenObservation(ObservationWrapper):
#     r"""Observation wrapper that flattens the observation."""
#
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = spaces.flatten_space(env.observation_space)
#
#     def observation(self, observation):
#         return np.concatenate((observation['image'].flatten(), observation['direction'].reshape(1), observation['mission']))

from gym.wrappers.flatten_observation import FlattenObservation

@hydra.main(version_base=None, config_path=memsearch.CONFIG_PATH, config_name="config")
def main(cfg):
    ray.init()

    def env_creator(env_config):
        return FlattenObservation(SMGFixedEnv(**env_config))

    register_env("SmgFixedEnv-v0", env_creator)

    scene_sampler, scene_evolver = make_scene_sampler_and_evolver_(cfg.scene_gen)
    env = FlattenObservation(SMGFixedEnv(
        scene_sampler=scene_sampler,
        scene_evolver=scene_evolver,
        encode_obs_im=True,
        mission_mode='one_hot'
    ))

    obs = env.reset()
    action = env.action_space.sample()
    all = env.step(action)
    breakpoint()


    config = (
        dqn.DQNConfig()
        .resources(num_gpus=1)
        .rollouts(num_rollout_workers=4, horizon=300)
        .framework("torch")
        .training(
            model={
                # Auto-wrap the custom(!) model with an LSTM.
                # "use_lstm": True,
                "framestack": True,
                # To further customize the LSTM auto-wrapper.
                # "lstm_cell_size": 64,
                # Specify our custom model from above.
                # Extra kwargs to be passed to your model's c'tor.
                # "custom_model_config": {},
            }
        )
        .environment(
            "SmgFixedEnv-v0",
            env_config={
                "scene_sampler": scene_sampler,
                "scene_evolver": scene_evolver,
                "encode_obs_im": True,
                "mission_mode": "one_hot",
            },
        )
    )

    config.replay_buffer_config.update(
        {
            "capacity": 10000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }
    )

    trainer = config.build()
    for _ in tqdm.tqdm(range(100000)):
        trainer.train()


if __name__ == "__main__":
    main()  # type: ignore
