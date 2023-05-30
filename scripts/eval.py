import os
import hydra
import logging
import numpy as np
import seaborn as sns
import memsearch
import functools
import multiprocessing
from tqdm import tqdm

from memsearch.util import configure_logging
from memsearch.metrics import AvgAccuracy, AvgAUC, DiscSumOfRewards, plot_agent_eval, store_agent_eval, rename
from memsearch.running import eval_agent
from memsearch.tasks import TaskType
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)

def eval_policies(cfg):
    logger = configure_logging(name='eval')
    task = TaskType(cfg.task.name)
    task_name = task.value
    final_averages = []
    if type(cfg.agents.agent_types) is str:
        agent_types = [cfg.agents.agent_types]
    else:
        agent_types = cfg.agents.agent_types
    output_dir = os.path.join(HydraConfig.get().runtime.output_dir, 'eval')
    images_dir = os.path.join(output_dir, 'images')
    metrics = [
        AvgAccuracy('%s_%s_avg_ac'%(cfg.run_name, task_name), images_dir),
        AvgAUC('%s_%s_avg_AuC'%(cfg.run_name,task_name), images_dir),
        DiscSumOfRewards('%s_%s_avg_DSoR'%(cfg.run_name, task_name), images_dir)
    ]

    if cfg.eval_in_parallel and len(agent_types) > 1:
        multiprocessing.set_start_method('spawn')
        run_eval_agent = functools.partial(eval_agent, cfg, logger, output_dir, task)
        tqdm.set_lock(multiprocessing.RLock())
        with multiprocessing.Pool(processes = len(agent_types)) as pool:
            all_score_vecs = pool.map(run_eval_agent, agent_types)
    else:
        all_score_vecs = []
        for agent_type in agent_types:
            logger.info('Evaluating %s'%agent_type)
            all_score_vecs.append(eval_agent(cfg, logger, output_dir, task, agent_type))

    sns.set()
    for i, agent_type in enumerate(agent_types):
        score_vecs = all_score_vecs[i]
        final_averages.append(np.mean(score_vecs))
        store_agent_eval(output_dir, score_vecs, agent_type)
        save_figs = (i == len(agent_types) - 1)
        plot_agent_eval(cfg.task.eps_per_scene, score_vecs, agent_type, i, metrics,
                        smoothing_kernel_size=cfg.task.num_smoothing_steps, task=task,
                        show_fig=False, save_fig=save_figs, x_labels=agent_types)

    for metric in metrics:
        logger.info('---')
        logger.info('Results for metric %s'%metric.get_metric_name())
        for i, agent_type in enumerate(cfg.agents.agent_types):
            logger.info('%s agent final average score: %.3f +/- %.3f'%(agent_type,
                                                                 metric.agent_evals[rename(agent_type)],
                                                                 metric.agent_evals_var[rename(agent_type)]))
        if 'upper_bound' in cfg.agents.agent_types:
            logger.info('---')
            logger.info('Results for metric %s normalized by upper bound'%metric.get_metric_name())
            for i, agent_type in enumerate(cfg.agents.agent_types):
                logger.info('%s agent final normalized average score: %.3f'%(agent_type,
                    metric.agent_evals[rename(agent_type)]/metric.agent_evals[rename('upper_bound')]))

@hydra.main(version_base=None, config_path=memsearch.CONFIG_PATH, config_name="config")
def main(cfg):
    eval_policies(cfg)

if __name__ == '__main__':
    main() #type: ignore
