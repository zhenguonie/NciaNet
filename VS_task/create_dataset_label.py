import os
import pickle
import shutil

data_folder = './original_data'
subfolders = ["training inactives", "training actives", "test actives", "test inactives"]
output_file = './data'
index_label_dict = {}
index_label_pairs = []

label_map = {
    "training inactives": 0,
    "training actives": 1,
    "test actives": 1,
    "test inactives": 0
}

for subfolder in subfolders:
    subfolder_path = os.path.join(data_folder, subfolder)
    label = label_map[subfolder]  
    if os.path.exists(subfolder_path):
        for folder in os.listdir(subfolder_path):
            folder_path = os.path.join(subfolder_path, folder)
            if os.path.isdir(folder_path):
                index_label_dict[folder] = label
    else:
        print(f"The path {subfolder_path} does not exist. Please check if the folder is correct!")

output_label_file = os.path.join(output_file, 'VS_task_label_dict.pkl')
with open(output_label_file, 'wb') as f:
    pickle.dump(index_label_dict, f)
print(f"Data has been successfully saved to {output_label_file}")

training_set_folder = os.path.join(output_file, "training set")
test_set_folder = os.path.join(output_file, "test set")

if not os.path.exists(training_set_folder):
    os.makedirs(training_set_folder)

if not os.path.exists(test_set_folder):
    os.makedirs(test_set_folder)

for subfolder in ["test actives", "test inactives"]:
    subfolder_path = os.path.join(data_folder, subfolder)
    if os.path.exists(subfolder_path):
        for folder in os.listdir(subfolder_path):
            folder_path = os.path.join(subfolder_path, folder)
            if os.path.isdir(folder_path):
                shutil.copytree(folder_path, os.path.join(test_set_folder, folder))

for subfolder in ["test actives", "test inactives"]:
    subfolder_path = os.path.join(data_folder, subfolder)

    if os.path.exists(subfolder_path):
        for folder in os.listdir(subfolder_path):
            folder_path = os.path.join(subfolder_path, folder)

            if os.path.isdir(folder_path):
                shutil.copytree(folder_path, os.path.join(training_set_folder, folder))

for subfolder in subfolders:
    subfolder_path = os.path.join(data_folder, subfolder)
    if os.path.exists(subfolder_path):
        for folder in os.listdir(subfolder_path):
            folder_path = os.path.join(subfolder_path, folder)
            if os.path.isdir(folder_path):
                if subfolder in ["training inactives", "training actives"]:
                    shutil.copytree(folder_path, os.path.join(training_set_folder, folder))


    else:
        print(f"The path {subfolder_path} does not exist. Please check if the folder is correct!")
        
print("The folders have been successfully copied to Test set and Training set, and the original folders have been deleted.")

data_path = './data/VS_task_label'
def load_label_data(data_path):
    with open(data_path, 'rb') as file:
        res = pickle.load(file)
    return res

label_data = load_label_data(output_label_file)
print(label_data)