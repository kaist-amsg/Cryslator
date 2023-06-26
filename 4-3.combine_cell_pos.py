import glob
import numpy as np
from ase.io import read,write
from ase import Atoms
import os
from tqdm import tqdm
import sys

def main():
    cell_folder = './cell_predicted_unrelaxed'
    #cell_folder = sys.argv[1]
    pos_folder = './pos_predicted_unrelaxed'
    #pos_folder = sys.argv[2]
    cif_save_folder = './translated_cif_xmno'
    os.makedirs(cif_save_folder,exist_ok=True)
    cif_folder = '../Cryslator/data/cifs_xmno'
    
    for cell_name in tqdm(glob.glob(cell_folder+'/*.npy')):
        pos_name = pos_folder+'/'+cell_name.split('\\')[-1].replace('_cell.npy','_pos.npy')
        cell = np.load(cell_name)
        pos = np.load(pos_name)
        cif_name = cell_name.split('\\')[-1].replace('_cell.npy','_unrelaxed.cif')
        atoms = read(cif_folder +'/'+ cif_name)
        atoms_pred = atoms.copy()
        atoms_pred.set_cell(cell)
        atoms_pred.set_scaled_positions(pos)
        atoms_pred.write(cif_save_folder+'/'+cif_name.replace('_unrelaxed.cif','_predicted.cif'))
    
    
    
if __name__ == "__main__":
    main()


