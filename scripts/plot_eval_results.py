from pathlib import Path
import numpy as np
from cycler import cycler
from scripts.eval import plot_agent_eval
from memsearch.metrics import AvgAccuracy, AvgAUC, DiscSumOfRewards, PercentObjectsFound, PercObjectsFoundOverTime, AvgNumAttempts
import argparse
import matplotlib.pyplot as plt
from memsearch.tasks import TaskType

def is_agent_name(test_str):
    ignore_keys = ['experiment=iclr', 'image', 'hydra']
    for key in ignore_keys:
        if key in test_str:
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
    )
    parser.add_argument(
        "--task",
        help="Name of the task to be evaluated.",
        default=None
    )
    parser.add_argument(
        "--metrics",
        default="all"
    )
    parser.add_argument(
        "--save_dir",
        help="Dir where generated plots are to be stored.",
        default=None
    )
    parser.add_argument(
        "--agent_order",
        help="Comma Separated Agent names, in the order to be plotted",
        #default="random,memorization,counts,sgm_heat,priors,upper_bound"
        default="random,counts,priors,memorization,sgm_heat,upper_bound"
    )
    parser.add_argument(
        "--num_smoothing_steps",
        help="Number of steps to use for smoothing for line graphs",
        default=10
    )
    
    args = parser.parse_args()
    log_dir = Path(f'outputs/{args.experiment_name}/eval')
    if args.task is None:
        if 'pl' in args.experiment_name:
            task = TaskType.PREDICT_LOC
        elif 'pls' in args.experiment_name:
            task = TaskType.PREDICT_LOC
        elif 'pd' in args.experiment_name:
            task = TaskType.PREDICT_ENV_DYNAMICS
        elif 'fo' in args.experiment_name:
            task = TaskType.FIND_OBJECT
        else:
            task = TaskType.PREDICT_LOC
    else:
        task = args.task
    if args.save_dir is None:
        save_dir = log_dir / 'images'
    else:
        save_dir = Path(args.save_dir)
    num_smoothing_steps = int(args.num_smoothing_steps)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics = [
            AvgAccuracy('%s_avg_ac'%task.value, save_dir, ymax=0.6),
            AvgAUC('%s_avg_AuC'%task.value, save_dir),
            DiscSumOfRewards('%s_avg_DSoR'%task.value, save_dir)
    ]
    """
    if args.metrics == 'all':
    elif args.metrics == 'line':
        metrics = [
                AvgAccuracy('%s_avg_ac'%task.value, save_dir, ymax=0.6)
        ]
        args.all_agent_order = "random,memorization,counts,sgm_heat,priors,upper_bound"
        plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'red', 'orange', 'purple','green', 'pink']))) 
    elif args.metrics == 'bar':
        metrics = [
                AvgAUC('%s_avg_AuC'%task.value, save_dir),
                DiscSumOfRewards('%s_avg_DSoR'%task.value, save_dir)
        ]
        args.agent_order = "random,counts,priors,memorization,sgm_heat,upper_bound"
        plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'orange', 'green', 'red', 'purple', 'pink']))) 
    """
    all_agent_names = args.agent_order.split(',')
    if task == 'find_object':
        metrics = [
            AvgAccuracy('%s_%s_avg_num_steps'%(args.experiment_name, task), save_dir, ymin=1.0, ymax=12, use_std=False),
            PercentObjectsFound('%s_%s_perc_found'%(args.experiment_name, task), save_dir, top_k=10),
            PercObjectsFoundOverTime('%s_%s_perc_found_over_time'%(args.experiment_name, task), save_dir, top_k=10),
            AvgNumAttempts('%s_%s_avg_num_attempts'%(args.experiment_name, task), save_dir)
        ]

    for i, agent_name in enumerate(all_agent_names):
        agent_csv_path = log_dir / agent_name
        csv_path = agent_csv_path / "eval.csv"
        agent_name = agent_csv_path.stem
        f = open(str(csv_path.resolve()), "r")
        score_vecs = [np.array(l.split(','), dtype=np.float32) for l in f.readlines()[-99:]]
        num_scenes, num_steps = len(score_vecs), score_vecs[0].shape[0]
        save_figs = (i == len(all_agent_names) - 1)
        plot_agent_eval(num_steps, score_vecs, agent_name, i, metrics,
                            smoothing_kernel_size=num_smoothing_steps, task=task, 
                            show_fig=False, save_fig=save_figs, x_labels=all_agent_names)
