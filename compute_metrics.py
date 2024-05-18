import argparse
import os
import json
import os
import json
import numpy as np
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Continual Learning with Test Time Adaptation")
    parser.add_argument("-jis", "--job_id_start", type=int, help="Enter jobid start")
    parser.add_argument("-jie", "--job_id_end", type=int, help="Enter jobid end")
    parser.add_argument("-dc", "--dataset_choice", type=str, help="Enter dataset for compilation of results")
    # parser.add_argument("-sp", "--save_path", type=str, help="Enter dataset for compilation of results")
    parser.add_argument("-sp", "--sparsity", type=str, help="Enter sparsity")
    args = parser.parse_args()

   

def build_stats(start_job_id, end_job_id, data_choice):
    # Initialize an empty dictionary to store stats
    compiled_data = {}
    stats_data = {}

    # Traverse the directory
    root_dir = "/home/vs2410/scratch/output"
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Check if the item in the directory is a folder and within the job_id range
        if os.path.isdir(folder_path):
            try:
                job_id = int(folder_name.split("_")[0])
               
                if start_job_id <= job_id <= end_job_id:
                    # Read the config.json file

                    all_files = os.listdir(folder_path)

                    # Filter files with .json extension
                    json_files = [file for file in all_files if file.endswith(".json")]

                    # Find the file that doesn't start with "config"
                    matrix_file = [file for file in json_files if not file.startswith("configs")]
                    matrix_file = os.path.join(folder_path,matrix_file[0])
                    
                    
                    with open(matrix_file, "r") as acc_matrices:
                        matrices_entry = json.load(acc_matrices)

                    student_matrix = matrices_entry["student_acc_matrix"]
                    teacher_matrix = matrices_entry["teacher_acc_matrix"]
                    acc_avg_student = np.mean(student_matrix[-1])
                    acc_avg_teacher = np.mean(teacher_matrix[-1])
                    
                    task_1_acc_student = -1
                    task_1_acc_teacher = -1
                    if student_matrix is not None:
                        task_1_acc_student = student_matrix[-1][0]
                    if teacher_matrix is not None:
                        task_1_acc_teacher = teacher_matrix[-1][0]
                    # print(task_1_acc_student, task_1_acc_teacher )
                    
                    config_path = os.path.join(folder_path, "configs.json")
                    with open(config_path, "r") as config_file:
                        config_data_entry = json.load(config_file)
                    
                    
                    # Extract information from config.json
                    current_data_choice = config_data_entry.get("dataset")
                    training_mode = config_data_entry.get("method")
                    # acc_avg_student = config_data_entry["train_config"].get("acc_avg_student")
                    # acc_avg_teacher = config_data_entry["train_config"].get("acc_avg_teacher")
                    # acc_avg = config_data_entry["train_config"].get("acc_avg") # CL 
                    # test_acc = config_data_entry["train_config"].get("test_acc") # joint
                    current_job_id = config_data_entry.get("job_id")
                    seed = config_data_entry.get("seed")
                    time_elapsed_hrs = config_data_entry.get("time_elapsed_hrs")
                    sp = config_data_entry.get("sparsity")


                    
                    # Check if the current data choice matches the specified one
                    if current_data_choice == data_choice and sp == sparsity:
                        # Create nested dictionary structure
                        if current_data_choice not in compiled_data:
                            compiled_data[current_data_choice] = {}
                        if training_mode not in compiled_data[current_data_choice]:
                            compiled_data[current_data_choice][training_mode] = {}
                
                        # Append accuracy metric to the list
                        if training_mode in ["SPU", "Finetune"]:
                            compiled_data[current_data_choice][training_mode][current_job_id] = {"acc_avg_student": acc_avg_student,"acc_avg_teacher":acc_avg_teacher, "acc_1st_task_student":task_1_acc_student, "acc_1st_task_teacher":task_1_acc_teacher }

            except Exception as e:
                # Ignore folders with non-integer names
                print(e)
                pass

    # Calculate average and standard deviation
    for dataset, modes in compiled_data.items():
        for mode, seed in modes.items():
            # print(mode, seed, seed.values())
            if mode=="joint":
                avg_accuracy = np.mean(list(seed.values()))
                std_dev_accuracy = np.std(list(seed.values()))
                if "avg" not in compiled_data[dataset][mode]:
                    compiled_data[dataset][mode]["avg"] = {}
                
                compiled_data[dataset][mode]["avg"] = {
                        'average_accuracy': avg_accuracy,
                        'std_dev_accuracy': std_dev_accuracy
                    }
            else:
                student_avg = []
                teacher_avg = []
                teacher_dict = {}
                student_1st_task = []
                teacher_1st_task = []

                for s, st in seed.items():
                    student_1st_task.append(st["acc_1st_task_student"])
                    teacher_1st_task.append(st["acc_1st_task_teacher"])

                    student_avg.append(st["acc_avg_student"])
                    teacher_avg.append(st["acc_avg_teacher"])

                student_avg_accuracy = np.mean(student_avg)
                student_std_dev_accuracy = np.std(student_avg)
                student_1st_task_avg = np.mean(student_1st_task)
                student_1st_task_std = np.std(student_1st_task)

                teacher_1st_task_avg = np.mean(teacher_1st_task)
                teacher_1st_task_std = np.std(teacher_1st_task)
  
                teacher_avg_accuracy = np.mean(teacher_avg)
                teacher_std_dev_accuracy = np.std(teacher_avg)
                teacher_dict = {
                'average_accuracy_teacher': teacher_avg_accuracy,
                'std_dev_accuracy_teacher': teacher_std_dev_accuracy
            }

                compiled_data[dataset][mode]["avg"] = { "student": {
                    'average_accuracy_student': student_avg_accuracy,
                    'std_dev_accuracy_student': student_std_dev_accuracy
                } , "teacher":  teacher_dict, 
                "task_1_stats": {"avg_student_1st_task": student_1st_task_avg , "std_dev_student_1st_task": student_1st_task_std, 
                                 "avg_teacher_1st_task": teacher_1st_task_avg , "std_dev_teacher_1st_task": teacher_1st_task_std,}
                
                }

        #     print()
        # print()
    return compiled_data

def save_stats(stats_data, output_file):
    # Save the stats data as a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(stats_data, json_file, indent=2)
        print(f"Stats data saved: {output_file}")

data_choice = args.dataset_choice
start_job_id = args.job_id_start
end_job_id = args.job_id_end
# save_path = args.save_path
sparsity = eval(args.sparsity)

result_stats = build_stats(start_job_id, end_job_id, data_choice)

# Save the stats data to a specified file
prefix_path =  "/home/vs2410/scratch/proj/FinetuneCLIP/compiled_results/17th_may/long_seq/"
suffix_path = data_choice+"_"+str(start_job_id)+"_"+str(end_job_id)
output_stats_file = prefix_path+suffix_path+"_stats.json"
save_stats(result_stats, output_stats_file)

#24409972 
#24410147

