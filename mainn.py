import os
import random
import time
from argparse import Namespace
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import wandb

from clip import clip
from dataset.aircraft import SplitAircraft
from dataset.birdsnap import SplitBirdsnap
from dataset.cars import SplitCars
from dataset.cifar100 import SplitCifar100
from dataset.cub import CUB
from dataset.gtsrb import SplitGTSRB
from trainer import METHOD
import json
from clip.clip import tokenize

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def save_config_write_to_file(save_dir_path, args):
    try:
        with open(os.path.join(save_dir_path, "configs.json"), "w") as fp:
            json.dump(vars(args), fp,)
        print(f"************** Contents will be saved at {save_dir_path} ")
    except Exception as error:
        print(error)


def get_time_stamp_for_saving_output(args):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")  # month/date/year_time
    print("date and time:", date_time)
    date_time = date_time.replace(", ", "_")
    date_time = date_time.replace("/", "_")
    args.timestamp = date_time
    if args.sanity:
        job_id = 280
    else:
        job_id = args.save_path.split("/")[-2].split(".")[-2]
    args.job_id = job_id
    dir_name = str(job_id) + "_" + str(date_time)
    save_dir_path = os.path.join(args.save_path, dir_name)

    try:
        os.mkdir(save_dir_path)
        print(f"************** Contents will be saved at {save_dir_path} ")
        args.save_path = save_dir_path
    except OSError as error:
        print(error)

    return save_dir_path


def plot_cl_metrics(args, average_accuracy_student, backward_transfer_student, acc_matrix_student=None, average_accuracy_teacher=None,
                    backward_transfer_teacher=None, acc_avg_student_lp=None, bt_student_lp=None, acc_matrix_teacher=None, acc_matrix_student_lp=None,
                    linear_probe=False, ):
    tasks = np.arange(1, args.num_tasks + 1)
    # Plot average accuracy
    # [acc_whole, acc_linear, acc_block]
    # [bt_whole, bt_linear, bt_block]
    plt.figure(figsize=(8, 4))
    plt.plot(tasks, average_accuracy_student, marker="o", color="green", label=f"Test Accuracy Student: {average_accuracy_student[-1]:.2f}", )
    plt.axhline(average_accuracy_student[-1], color="green", linestyle="--")

    if average_accuracy_teacher is not None:
        plt.plot(tasks, average_accuracy_teacher, marker="o", color="red", label=f"Test Accuracy Teacher: {average_accuracy_teacher[-1]:.2f}", )
        plt.axhline(average_accuracy_teacher[-1], color="red", linestyle="--")

    if linear_probe:
        plt.plot(tasks, acc_avg_student_lp, marker="o", color="blue", label=f"LP Accuracy Student: {acc_avg_student_lp[-1]:.2f}", )
        plt.axhline(acc_avg_student_lp[-1], color="blue", linestyle="--")

    plt.xlabel("Task")
    plt.ylabel("Average Accuracy")
    plt.title(f'Average Accuracy per Task: jobid-  {args.job_id}')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(args.save_path, str(args.timestamp + "_cl_average_accuracy_per_task.png"), )
    plt.savefig(acc_path, bbox_inches="tight")
    # plt.show()

    # Plot backward transfer
    plt.figure(figsize=(8, 4))
    plt.plot(tasks[1:], backward_transfer_student, marker="o", color="green", label="Student")

    if backward_transfer_teacher is not None:
        plt.plot(tasks[1:], backward_transfer_teacher, marker="o", color="red", label="Teacher")
    if linear_probe:
        plt.plot(tasks[1:], bt_student_lp, marker="o", color="blue", label="Student LP")

    plt.xlabel("Task")
    plt.ylabel("Backward Transfer")
    plt.title(f'Backward Transfer per Task: jobid-  {args.job_id}')
    plt.grid(True)
    plt.legend()

    bt_path = os.path.join(args.save_path, str(args.timestamp + "_cl_backward_transfer.png"), )
    plt.savefig(bt_path, bbox_inches="tight")
    # plt.show()

    ## Plot 1st task accuracies
    plt.figure(figsize=(8, 4))
    tasks = np.arange(1, args.num_tasks + 1)
    student_1st_task_acc = np.array([inner[0] for inner in acc_matrix_student])

    if len(acc_matrix_teacher)==args.num_tasks:
        teacher_1st_task_acc = np.array([inner[0] for inner in acc_matrix_teacher])
        plt.plot(tasks, teacher_1st_task_acc, marker="o", color="red", label=f"Teacher: {teacher_1st_task_acc[-1]:.2f}", )
    plt.plot(tasks, student_1st_task_acc, marker="o", color="green", label=f"Student: {student_1st_task_acc[-1]:.2f}", )
    if linear_probe:
        plt.plot(tasks, np.array([inner[0] for inner in acc_matrix_student_lp]), marker="o", color="blue", label="Student_LP", )

    plt.xlabel("Task")
    plt.ylabel("Accuracy of 1st Task")
    plt.title(f'Accuracy of 1st Task: jobid-  {args.job_id}')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(args.save_path, str(args.timestamp + "_1st_task_accuracy.png"), )
    plt.savefig(acc_path, bbox_inches="tight")

    ## Plot 2nd task accuracies
    plt.figure(figsize=(8, 4))
    tasks = np.arange(2, args.num_tasks + 1)
    if len(acc_matrix_teacher)==args.num_tasks:
        plt.plot(tasks, np.array([inner[1] for inner in acc_matrix_teacher[1:]]), marker="o", color="red", label="Teacher", )
    plt.plot(tasks, np.array([inner[1] for inner in acc_matrix_student[1:]]), marker="o", color="green", label="Student", )
    if linear_probe:
        plt.plot(tasks, np.array([inner[1] for inner in acc_matrix_student_lp[1:]]), marker="o", color="blue", label="Student_LP", )

    plt.xlabel("Task")
    plt.ylabel("Accuracy of 2nd Task")
    plt.title(f'Accuracy of 2nd Task: jobid-  {args.job_id}')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(args.save_path, str(args.timestamp + "_2nd_task_accuracy.png"), )
    plt.savefig(acc_path, bbox_inches="tight")

    ## Plot 3rd task accuracies
    plt.figure(figsize=(8, 4))
    tasks = np.arange(3, args.num_tasks + 1)
    if len(acc_matrix_teacher)==args.num_tasks:
        plt.plot(tasks, np.array([inner[2] for inner in acc_matrix_teacher[2:]]), marker="o", color="red", label="Teacher", )
    plt.plot(tasks, np.array([inner[2] for inner in acc_matrix_student[2:]]), marker="o", color="green", label="Student", )
    if linear_probe:
        plt.plot(tasks, np.array([inner[2] for inner in acc_matrix_student_lp[2:]]), marker="o", color="blue", label="Student_LP", )

    plt.xlabel("Task")
    plt.ylabel("Accuracy of 3rd Task")
    plt.title(f'Accuracy of 3rd Task: jobid-  {args.job_id}')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(args.save_path, str(args.timestamp + "_3rd_task_accuracy.png"), )
    plt.savefig(acc_path, bbox_inches="tight")

    # saving acc matrix
    mat_path = os.path.join(args.save_path, str(args.timestamp + "acc_matrices.json"), )
    data = {"student_acc_matrix": acc_matrix_student, "teacher_acc_matrix": acc_matrix_teacher, "LP": acc_matrix_student_lp}
    import json
    with open(mat_path, 'w') as f:
        json.dump(data, f)


def calculate_metrics(accuracy_matrix):
    num_tasks = len(accuracy_matrix)

    # Calculate average accuracy
    average_accuracy = [sum(i) / (j+1) for j, i in enumerate(accuracy_matrix)]
    print(average_accuracy)
    # average_accuracy = sum(accuracy_matrix[i] for i in range(current_task))/current_task

    # Calculate backward transfer
    backward_transfer = np.zeros(num_tasks - 1)
    for i in range(1, num_tasks):
        for j in range(i):
            backward_transfer[i - 1] = accuracy_matrix[i][j] - accuracy_matrix[i - 1][j]
    print(backward_transfer)

    return average_accuracy, backward_transfer


def seed_everything(seed=1234):
    print(f"************* seed is {seed} **************")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)   # it doesnt work with vit trainable backbone. Some operations are non-det.


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(args):
    args = omegaconf.OmegaConf.to_container(args)
    args = Namespace(**args)
    seed_everything(args.seed)
    
    start = time.time()
    # random_seed(args.seed)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # get the name of the experiments and setup logging
    if args.name is None:
        args.name = '-'.join([args.method, args.dataset, os.environ.get("SLURM_JOB_ID", ""), ])
    log_base_path = os.path.join(args.logs, args.name)

    os.makedirs(log_base_path, exist_ok=True)
    args.log_path = log_base_path
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    if args.wandb:

        wandb.init(# put your wandb initiation

        )

    save_dir_path = get_time_stamp_for_saving_output(args)

    # set up model
    model, transform = clip.load(args.model, download_root='./clip_models/', args=args)

    teacher_model, _ = clip.load(args.model, download_root='./clip_models/', args=args)

    args.hidden_size = model.visual.proj.shape[0]
    args.visual_layers = len(model.visual.transformer.resblocks)

    if args.dataset == 'cifar100':
        dataset = SplitCifar100(args, args.data, transform)
    elif args.dataset == 'cars':
        dataset = SplitCars(args, transform=transform)
    elif args.dataset == 'cub':
        dataset = CUB(args, transform=transform)
    elif args.dataset == 'aircraft':
        dataset = SplitAircraft(args, transform=transform)
    elif args.dataset == 'birdsnap':
        dataset = SplitBirdsnap(args, transform=transform)
    elif args.dataset == 'gtsrb':
        dataset = SplitGTSRB(args, transform=transform)
    else:
        raise ValueError

    args.num_classes = dataset.num_classes
    args.num_tasks = dataset.num_tasks
    args.scenario = dataset.scenario
    Trainer = METHOD[args.method](args)
    if args.sanity:
        args.batch_size=2

    save_config_write_to_file(save_dir_path, args)

    acc_matrix_teacher = []
    acc_matrix_student = []

    for task in range(dataset.num_tasks):
        if args.sweep and task == 3:
            break
        print(f'Train task {task}')
        if args.evaluation:
            Trainer.only_evaluation(model, dataset, task, acc_matrix=acc_matrix_student)
            continue
        Trainer.train(teacher_model, model, dataset, task)
        if not args.ema and task>0:
            teacher_model.load_state_dict(model.state_dict())
        if args.tta_phase and task >= 0:
            print("------------------- Evaluation of Student model before TTA ----------------------")
            Trainer.evaluation(model, dataset, task, acc_matrix=acc_matrix_student)

            print("------------------- Evaluation of Teacher model before TTA ----------------------")
            Trainer.evaluation(teacher_model, dataset, task, acc_matrix=acc_matrix_teacher)

            Trainer.tta_with_merged_data(teacher_model, model, dataset, task)
        print("------------------- Evaluation of Student model ----------------------")
        Trainer.evaluation(model, dataset, task, acc_matrix=acc_matrix_student)

        print("------------------- Evaluation of Teacher model ----------------------")
        Trainer.evaluation(teacher_model, dataset, task, acc_matrix=acc_matrix_teacher)

        Trainer.save_checkpoint(model, task, args)

        if task==2 and args.sanity:
            args.num_tasks=3
            break

    print(f'Total training time in hours: {(time.time() - start) / 3600: .3f}')

    print("{}".format(args).replace(', ', ',\n'))

    if args.wandb:
        wandb.finish()

    acc_avg_teacher, bt_teacher = calculate_metrics(acc_matrix_teacher)
    acc_avg_student, bt_student = calculate_metrics(acc_matrix_student)
    plot_cl_metrics(args, acc_avg_student, bt_student, acc_matrix_student, average_accuracy_teacher=acc_avg_teacher,
                    backward_transfer_teacher=bt_teacher, acc_matrix_teacher=acc_matrix_teacher)


if __name__ == '__main__':
    main()
