import numpy as np
import argparse


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# continual learning metric
class ClassIncrementalMetric(object):
    """Computes and stores the average and current value for model at every learning time"""

    def __init__(self, args):
        self.task = args.num_tasks
        self.scenario = args.scenario
        assert (self.scenario in ['class_incremental', 'dataset_incremental'])
        self.reset()
        self.set_task()

    def reset(self):
        self.average_accuracy = [0] * self.task
        self.matrix = np.zeros([self.task, self.task])
        self.current_last_task = 0
        self.forgetting = np.zeros(self.task)
        self.learning = np.zeros(self.task)

    def set_task(self):
        self.metrics = {}
        self.per_timestep_metric = {}
        for t in range(self.task):
            self.metrics[t] = AverageMeter()
            self.per_timestep_metric[t] = {}
            for tt in range(self.task):
                self.per_timestep_metric[t][tt] = AverageMeter()

    def update(self, model_no, data_no, val, n=1):
        self.metrics[model_no].update(val, n)
        self.per_timestep_metric[model_no][data_no].update(val, n)

    def update_metric(self, model_no, data_no):
        self.matrix[model_no,
                    data_no] = self.per_timestep_metric[model_no][data_no].avg
        self.average_accuracy[model_no] = self.metrics[model_no].avg
        if model_no == data_no and data_no > 0:
            m = self.matrix[:model_no + 1, :model_no + 1]
            mm = np.max(m[:-1], axis=0)
            self.forgetting[model_no] = np.mean(mm - m[-1])
        if model_no == data_no:
            self.learning[model_no] = self.matrix[model_no, data_no]

    def print_matrix(self, name):
        print(f'Metric {name}')
        with np.printoptions(precision=3, suppress=True):
            print(self.matrix)


class TaskIncrementalMetric(object):
    """Computes and stores the average and current value for model at every learning time"""

    def __init__(self, args):
        self.task = args.num_tasks
        self.scenario = args.scenario
        self.reset()
        self.set_task()

    def reset(self):
        self.average_accuracy = [0] * self.task
        self.forgetting = [0] * self.task
        self.learning_average = [0] * self.task
        self.matrix = np.zeros([self.task, self.task])
        self.learning = np.zeros(self.task)

    def set_task(self):
        self.metrics = {}
        for t in range(self.task):
            self.metrics[t] = {}
            for dt in range(self.task):
                self.metrics[t][dt] = AverageMeter()

    def update(self, model_no, data_no, val, n=1):
        self.metrics[model_no][data_no].update(val, n)

    def update_metric(self, model_no, data_no):
        self.matrix[model_no, data_no] = self.metrics[model_no][data_no].avg
        self.average_accuracy[model_no] = np.mean(
            self.matrix[model_no, :model_no + 1])
        self.learning_average[model_no] = np.mean(
            np.diag(self.matrix[:model_no + 1]))
        if model_no == data_no and data_no > 0:
            m = self.matrix[:model_no + 1, :model_no + 1]
            mm = np.max(m[:-1], axis=0)
            self.forgetting[model_no] = np.mean(mm - m[-1])
        if model_no == data_no:
            self.learning[model_no] = self.matrix[model_no, data_no]

    def print_matrix(self, name):
        print(f'Metric {name}')
        with np.printoptions(precision=3, suppress=True):
            print(self.matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_tasks = 5
    args.scenario = 'class_incremental'
    continual_metric = ClassIncrementalMetric(args)
    continual_zero_shot_metric = ClassIncrementalMetric(args)

    for model in range(5):
        # training goes here
        # test phase
        for data in range(5):

            if data <= model:
                # continual learning evaluation goes here
                # classification space is the seen classes space
                for batch in range(1):
                    continual_metric.update(model, data, 0.8, 1)
                continual_metric.update_metric(model, data)
            # continual learning zero-shot evaluation goes here
            # classification space is the whole classes space
            for batch in range(1):
                continual_zero_shot_metric.update(model, data, 0.6, 1)
            continual_zero_shot_metric.update_metric(model, data)

        print(
            f'Accuracy at task {model}: {continual_metric.average_accuracy[model]: .3f}')
        print(
            f'GZSL Accuracy at task {model}: {continual_zero_shot_metric.average_accuracy[model]: .3f}')
    continual_metric.print_matrix('Continual Learning Accuracy')
    continual_zero_shot_metric.print_matrix('Generalized Zero-shot Accuracy')
