# Cryslator


Implementation of Cryslator developed by Prof. Yousung Jung group at Seoul National University (contact: yousung@gmail.com)

![figure1](https://github.com/kaist-amsg/Cryslator/assets/38600256/8e834b3f-9010-4fc2-936c-febb381f2dd2)

# Developer
Sungwon Kim (syaym@kaist.ac.kr)

# How to Use
First, take jsons_xmno_rcut6 from (X-Mn-O pair dataset), and execute the codes in numerical order.

1.train_GCN.py : Trainig Graph-Encoder and Regressor

2.train-Cryslator.py : Training Cryslator which predicts relaxed crystal feature from unrelaxed crystal feature

3-1.train_structure_predictor_cell.py & 3-2.train_structure_predictor_pos: Training Structure_Predictor predicting unit cell and coordinate changes before and after structural relaxation.

4-1.predict_cell.py & 4-2.predict_pos.py & 4-3.combine_cell_pos.py : Predicting the relaxed geometry of x-mn-o testset

4-4.check_atoms.py : Comparing predicted relaxed structure and groundtruth relaxed structures
 
