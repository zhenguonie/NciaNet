## NciaNet-Paddle
Here is the Source code of binding affinity (BA) task for paper: "NciaNet: A Non-covalent interaction-aware Graph Neural Network for the Prediction of Protein-Ligand Interaction in Drug Discovey".

Additionally, regarding the source code of virtual-screening (VS) task is put in the folder of ./VS_task

### Dependencies
- python >= 3.8
- paddlepaddle >= 2.1.0
- pgl >= 2.1.4
- openbabel == 3.1.1 (optional, only for preprocessing)

### Datasets and Preparation
The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).

You may need to use the [UCSF Chimera tool](https://www.cgl.ucsf.edu/chimera/) to convert the PDB-format files into MOL2-format files for feature extraction at first.

Alternatively, we also provided a [One Drive link](https://1drv.ms/f/s!Ap_z1OHP_xEagUyGOLgKARDNHw5b?e=kJW9Vp) for downloading converted PDBbind datasets.

You can also use the processed data from [One Drive link](https://1drv.ms/u/s!Ap_z1OHP_xEagUfpFIT1g51lMzcE?e=TbK2co). Before training the model, please put the downloaded files into the directory (./data/).

### Biochemical information
The cauculation of length and angle of hydrogen bond provided by [here](https://github.com/psa-lab/Hbind).

Hbond.zip deposits the data about the calculated angles and lengths of hydrogen bonds.Before training the model, please manually uncompress this folder.Then copy these to the .Hbond/refined-set2 folder

### How to run
To train the model, please run ./train.py file, then representing the result of four metric (RMSE/R/MAE/SD).

To Test the model, please run ./test.py

### Result
The result automatically saved in the ./output floder as result.txt.

### Configuration of Key Hyper-parameter 
cutoff 1 in 533 of ./process_pdbbind.py, default=9

cutoff 2 in 408 of ./process_pdbbind.py, default=5

Heads of multi-attention in 112 of ./layers.py, default=4


