import matplotlib.pyplot as plt
import numpy as np
from memsearch.tasks import TaskType
from abc import abstractmethod
from enum import Enum
from memsearch.tasks import TaskType

def rename(agent_type):
    if agent_type == 'counts':
        agent_type = 'Frequentist'
    elif agent_type == 'memorization':
        agent_type = 'Myopic'
    elif agent_type == 'upper_bound':
        agent_type = 'Oracle'
    elif agent_type == 'sgm_heat':
        agent_type = 'NES'
    else:
        agent_type = agent_type.capitalize()
    return agent_type

AGENT_COLORS = {
    'Random': 'brown',
    'Priors': 'orange',
    'Frequentist': 'red',
    'Myopic': 'green',
    'Oracle': 'purple',
    'SGM': 'blue',
}

class PlotType(Enum):
    LINE = "line"
    BAR = "bar"

class Metric(object):
    metric_id = 0

    def __init__(self, name, save_dir, plot_type):
        Metric.metric_id += 1
        self.fig_num = Metric.metric_id
        self.name = name
        self.plot_type = plot_type
        self.save_dir = save_dir
        self.agent_evals = {}
        self.agent_evals_var = {}

    @abstractmethod
    def add_data_to_plot(self, x, y, y_std, label, log_f, **kwargs):
        pass

    @abstractmethod
    def make_plot(self, agent_types, task, **kwargs):
        pass

    @abstractmethod
    def get_metric_name(self):
        pass

    def save_plot(self, save_name=None):
        if save_name is None:
            save_name = self.name
        plt.figure(self.fig_num)
        plt.savefig(f'{self.save_dir}/{save_name}.png')

    def show_plot(self):
        plt.figure(self.fig_num)
        plt.show()

    def add_data_to_csv(self, log_f, data):
        data = ["{:.3f}".format(data_i) for data_i in data]
        log_str = ",".join(data) + '\n'
        log_f.write(log_str)

class AvgAccuracy(Metric):
    def __init__(self, name, save_dir, ymin=0, ymax=1.0, use_std=True):
        super().__init__(name, save_dir, PlotType.LINE)
        self.ymin = ymin
        self.ymax = ymax
        self.use_std = use_std

    def get_metric_name(self):
        return 'average_accuracy'

    def add_data_to_plot(self, x, y, y_std, label):
        plt.figure(self.fig_num)#, figsize=(12.0,8.5))
        dy = (y_std**2)
        plt.plot(x, y, label=label)
        if self.use_std:
            plt.fill_between(x, y - dy, y + dy, alpha=0.2)
        self.agent_evals[label] = np.mean(y)
        self.agent_evals_var[label] = np.mean(dy)

    def make_plot(self, agent_types, task):
        plt.figure(self.fig_num)#, figsize=(12.0,8.5))
        plt.ylim(self.ymin, self.ymax)
        if task == TaskType.FIND_OBJECT:
            plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5),ncol=1, fancybox=True, shadow=True)
            plt.title('Number of Tries to Find the Object vs Step')
        else:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98),ncol=3, fancybox=True, shadow=True)
            plt.title('Predict Location Average Accuracy vs Step')
        plt.tight_layout()

class AvgAUC(Metric):
    def __init__(self, name, save_dir):
        super().__init__(name, save_dir, PlotType.BAR)

    def get_metric_name(self):
        return 'average_auc'

    def add_data_to_plot(self, x, y, y_std, label):
        plt.figure(self.fig_num)
        if isinstance(y, np.ndarray):
            auc = np.sum(y)
            auc_std = np.sqrt(np.sum(y_std**2))
        else:
            auc = y
            auc_std = y_std
        plt.bar(x, auc, yerr=auc_std, align='center', alpha=0.5, ecolor='black', capsize=10, label=label)
        self.agent_evals[label] = auc/100.0
        self.agent_evals_var[label] = np.mean(auc_std)/100.0

    def make_plot(self, agent_types, task):
        plt.figure(self.fig_num)
        plt.title('Predict Location Overall Average Accuracy')
        plt.xticks(np.arange(len(agent_types)), labels=agent_types)
        plt.grid(visible=True, axis='y')
        # set x spacing so that labels dont overlap
        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2 # inch margin
        s = maxsize/plt.gcf().dpi*len(agent_types)+2*m
        margin = m/plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        plt.gcf().tight_layout()

class DiscSumOfRewards(Metric):
    def __init__(self, name, save_dir):
        self.plot_type = PlotType.BAR
        super().__init__(name, save_dir, PlotType.BAR)

    def get_metric_name(self):
        return 'average_disc_sum_rewards'

    def add_data_to_plot(self, x, y, y_std, label):
        plt.figure(self.fig_num)
        discount_fac = 0.99
        if isinstance(y, np.ndarray):
            discount_coeffs = [discount_fac**i for i in range(1, len(y)+1)]
            disc_sum = np.dot(discount_coeffs, y)
            disc_sum_std = np.sqrt(np.dot(discount_coeffs, y_std**2))
        else:
            disc_sum = y
            disc_sum_std = y_std
        plt.bar(x, disc_sum, yerr=disc_sum_std, align='center', alpha=0.5, ecolor='black', capsize=10, label=label)
        self.agent_evals[label] = disc_sum
        self.agent_evals_var[label] = disc_sum_std

    def make_plot(self, agent_types, task):
        plt.figure(self.fig_num)
        plt.title('Discounted Sum of Rewards')
        plt.xticks(np.arange(len(agent_types)), labels=agent_types)
        plt.grid(visible=True, axis='y')
        # set x spacing so that labels dont overlap
        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2 # inch margin
        s = maxsize/plt.gcf().dpi*len(agent_types)+2*m
        margin = m/plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        plt.gcf().tight_layout()

class PercentObjectsFound(Metric):
    def __init__(self, name, save_dir, top_k):
        super().__init__(name, save_dir, PlotType.BAR)
        self.top_k = top_k

    def get_metric_name(self):
        return 'percent_objects_found'

    def add_data_to_plot(self, x, score_matrix, label):
        plt.figure(self.fig_num)
        num_steps = score_matrix.shape[1]
        num_scenes = score_matrix.shape[0]
        y = np.array(
            [np.count_nonzero(scene_scores != self.top_k+1) for scene_scores in score_matrix]
        )
        percent_found = np.sum(y) / (num_scenes * num_steps)
        plt.bar(x, percent_found, align='center', alpha=0.5, ecolor='black', capsize=10, label=label)
        self.agent_evals[label] = percent_found
        self.agent_evals_var[label] = 0.0

    def make_plot(self, agent_types, task):
        plt.figure(self.fig_num)
        plt.title('Percent Objects Found')
        plt.xticks(np.arange(len(agent_types)), labels=agent_types)
        plt.grid(visible=True, axis='y')
        # set x spacing so that labels dont overlap
        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2 # inch margin
        s = maxsize/plt.gcf().dpi*len(agent_types)+2*m
        margin = m/plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        plt.tight_layout()

class PercObjectsFoundOverTime(Metric):
    def __init__(self, name, save_dir, top_k):
        super().__init__(name, save_dir, PlotType.LINE)
        self.top_k = top_k

    def get_metric_name(self):
        return 'percent_objects_found'

    def add_data_to_plot(self, x, score_matrix, label):
        plt.figure(self.fig_num)#, figsize=(12.0,8.5))
        num_steps = score_matrix.shape[1]
        num_scenes = score_matrix.shape[0]
        y_object_found_arr = (score_matrix != self.top_k+1)
        y = np.array(
            [np.count_nonzero(y_object_found_arr[:, step_i]) / num_scenes for step_i in range(num_steps)]
        )
        print(y)
        y_std = np.std(y_object_found_arr, axis=0)
        dy = (y_std**2)
        plt.plot(x, y, label=label)
        # plt.fill_between(x, y - dy, y + dy, alpha=0.2)
        self.agent_evals[label] = np.mean(y)
        self.agent_evals_var[label] = np.mean(dy)

    def make_plot(self, agent_types, task):
        plt.figure(self.fig_num, figsize=(12.0,25.0))
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
          ncol=1, fancybox=True, shadow=True)
        plt.ylim(0,1.0)
        plt.title('Percent Objects Found vs Step')
        plt.tight_layout()

class AvgNumAttempts(Metric):
    def __init__(self, name, save_dir):
        super().__init__(name, save_dir, PlotType.BAR)

    def get_metric_name(self):
        return 'avg_num_attempts'

    def add_data_to_plot(self, x, score_matrix, label):
        plt.figure(self.fig_num)
        avg_num_steps = np.mean(score_matrix)
        plt.bar(x, avg_num_steps, align='center', alpha=0.5, ecolor='black', capsize=10, label=label)
        self.agent_evals[label] = avg_num_steps
        self.agent_evals_var[label] = 0.0

    def make_plot(self, agent_types, task):
        plt.figure(self.fig_num)
        plt.title('Average Number of Attempts')
        plt.xticks(np.arange(len(agent_types)), labels=agent_types)
        plt.grid(visible=True, axis='y')
        # set x spacing so that labels dont overlap
        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2 # inch margin
        s = maxsize/plt.gcf().dpi*len(agent_types)+2*m
        margin = m/plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        plt.tight_layout()

def plot_agent_eval(num_steps,
                    score_vecs,
                    agent_type,
                    x_ind,
                    metrics,
                    smoothing_kernel_size=10,
                    task=TaskType.PREDICT_LOC,
                    show_fig=False,
                    save_fig=False,
                    x_labels=None):
    """
    Adds plot information for each agent to the figure.
    Makes the final plot whenever show_fig or save_fig are turned on.

    args:
    num_steps: Number of steps the agent was evaluated for
    score_vecs: Score vectors containing performance metrics for the agent
    agent_type: Name of the agent
    plt_axs: Matplotlib figure axes on which plots are to be made. Currently configured for a figure with 3 subplots.
    x_ind: x index to be used for this agent when making bar plots
    metrics: metrics to plot for each agent. Must be an instance of the Metrics class
    show_fig: To be turned on if a figure is only to be shown
    save_fig: To be turn on if a figure is to be saved
    plot_args: List [cfg, agent_types] to be used for making the final plot
    """
    if smoothing_kernel_size!=0:
        kernel = np.ones(smoothing_kernel_size) / float(smoothing_kernel_size)
        smoothed_score_vecs = []
        for score_vec in score_vecs:
            smoothed_score_vec = np.convolve(score_vec, kernel, mode='valid')
            smoothed_score_vecs.append(smoothed_score_vec)

        x = np.array(range(num_steps-smoothing_kernel_size+1))
        score_matrix = np.stack(smoothed_score_vecs)
    else:
        x = np.array(range(num_steps))
        score_matrix = np.stack(score_vecs)
    y = np.mean(score_matrix, axis=0)
    y_std = np.std(score_matrix, axis=0)

    x_labels = x_labels[:]
    for i,agent_t in enumerate(x_labels):
        x_labels[i] = rename(agent_t)
    agent_type = rename(agent_type)

    for metric in metrics:
        if type(metric) is AvgAccuracy:
            metric.add_data_to_plot(x, y, y_std, agent_type)
        elif type(metric) in [PercentObjectsFound, AvgNumAttempts]:
            metric.add_data_to_plot(x_ind, score_matrix, agent_type)
        elif type(metric) is PercObjectsFoundOverTime:
            metric.add_data_to_plot(x, score_matrix, agent_type)
        else:
            metric.add_data_to_plot(x_ind, y, y_std, agent_type)

    if show_fig or save_fig:
        # Final Plot
        for metric in metrics:
            metric.make_plot(x_labels, task)
            if save_fig:
                metric.save_plot()
            if show_fig:
                plt.show()

def store_agent_eval(log_dir, scores, agent_type):
    log_csv_path = f'{log_dir}/{agent_type}/eval.csv'
    with open(log_csv_path, 'a') as csv_file:
        for eval_run in scores:
            eval_run_str = ["{:.3f}".format(score) for score in eval_run]
            log_str = ",".join(eval_run_str)+'\n'
            csv_file.write(log_str)
