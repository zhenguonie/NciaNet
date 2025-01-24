## NciaNet-Paddle
Source code of virtual-screening (VS) task in the file of vs_task for paper: "NciaNet: A Non-covalent interaction-aware Graph Neural Network for the Prediction of Protein-Ligand Interaction in Drug Discovey".

### Dependencies
- python >= 3.8
- paddlepaddle >= 2.1.0 
- pgl >= 2.1.4
- openbabel == 3.1.1 (optional, only for preprocessing)

### Datasets
Training set refer to papaer [here](https://link.springer.com/article/10.1186/s13321-024-00912-2).
The LIT-PCBA dataset can be downloaded [here](https://drugdesign.unistra.fr/LIT-PCBA/).

### Preparation of Biochemical information
The cauculation of length and angle of hydrogen bond provided by [here](https://github.com/psa-lab/Hbind).

Length and angle of hyrogen bond put in ./Hbond/LIT-PCBA/ according to index,as same as binding affinity task.

### Data Preparation
Fitstly, put the individual sample file named by user-defined index, including .PDB and .mol2 format, correspondingly put into the sub-folders of ./original_data folder. Original_data folder have 4 type sub-folders (training activces/training inactives/test actives/test inactives).

Then, run ./create_dataset_label.py files to creat the dictionary of indexs of samples and corresponding labels (actives as 1, inactives 0), outputing an VS_task_label_dict.pkl in ./data folder. In the meantime, original data files in ./original_data is re-arranged to form training set folder and test folder in ./data folder. 

Subsequently, run the ./preprocess_LIT_PCBA.py file to finish the data prepocessing, which saved into the ./data folder.

### How to run
To train the model, please run ./train.py file, then representing the result of EF 1% metric.

### Result
the result automatically saved in the ./output floder as result.txt.

### Configuration of Key Hyper-parameter 
cutoff 1 in 528 of ./preprocess_LIT_PCBA.py, default=9

cutoff 2 in 401 of ./preprocess_LIT_PCBA.py, default=5

Heads of multi-attention in 105 of ./layers.py, default=4

