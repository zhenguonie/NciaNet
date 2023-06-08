## NciaNet-Paddle
Prototype and draft of source code for paper: "NciaNet: A Non-covalent interaction-aware Graph Neural Network for the Prediction of Protein-Ligand Interaction in Drug Discovey".
To make it easier to understand, we will gradually optimize the construction and representation of the codes and add annotations in the near future

### Dependencies
- python >= 3.8
- paddlepaddle >= 2.1.0
- pgl >= 2.1.4
- openbabel == 3.1.1 (optional, only for preprocessing)

### Datasets
The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).

You may need to use the [UCSF Chimera tool](https://www.cgl.ucsf.edu/chimera/) to convert the PDB-format files into MOL2-format files for feature extraction at first.

Alternatively, we also provided a [One Drive link](https://1drv.ms/f/s!Ap_z1OHP_xEagUyGOLgKARDNHw5b?e=kJW9Vp) for downloading converted PDBbind datasets.

You can also use the processed data from [One Drive link](https://1drv.ms/u/s!Ap_z1OHP_xEagUfpFIT1g51lMzcE?e=TbK2co). Before training the model, please put the downloaded files into the directory (./data/).

### Biochemical information
Hbond folder deposits the data about the calculated angles and lengths of hydrogen bonds
### How to run
To train the model, you can run train.py
