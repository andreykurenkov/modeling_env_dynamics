import hydra

import memsearch
from memsearch.igridson_env import SMGFixedEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
# from ray.rllib.algorithms.ppo.ppo import PPOConfig, PPO
import tqdm
from memsearch.igridson_utils import make_scene_sampler_and_evolver_
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from memsearch.rl.complex_input_network import ComplexInputNetwork


@hydra.main(version_base=None, config_path=memsearch.CONFIG_PATH, config_name="config")
def main(cfg):
    def env_creator(env_config):
        return SMGFixedEnv(**env_config)

    register_env("SmgFixedEnv-v0", env_creator)
    ModelCatalog.register_custom_model("ComplexInputNetwork", ComplexInputNetwork)

    scene_sampler, scene_evolver = make_scene_sampler_and_evolver_(cfg.scene_gen)
    config = (
        DQNConfig()
        .resources(num_gpus=1)
        .rollouts( num_rollout_workers=4, horizon=300)
        .framework("torch")
        .training(
            model={
                "custom_model": "ComplexInputNetwork",
                "conv_filters": [[32, 4, 4], [64, 2, 2]]
            }
        )
        .environment(
            "SmgFixedEnv-v0",
            env_config={
                "scene_sampler": scene_sampler,
                "scene_evolver": scene_evolver,
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

    trainer = DQN(config=config)
    for _ in tqdm.tqdm(range(100000)):
        trainer.train()


if __name__ == "__main__":
    main()  # type: ignore
