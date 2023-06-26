from ase.io import read,write
from ase import Atoms
import glob
from pymatgen.io.ase import *
from pymatgen.analysis.structure_matcher import StructureMatcher,ElementComparator
from tqdm import tqdm
import sys
import json

def main():
    cif_folder1 = '../Cryslator/data/cifs_xmno'
    #cif_folder2 = './cifs_save_xmno'
    cif_folder2 = './translated_cif_xmno'
    
    sm = StructureMatcher(comparator=ElementComparator(),ltol=0.2,stol=0.3,angle_tol=5,primitive_cell=True)
    sm2  = StructureMatcher(comparator=ElementComparator(),ltol=1,stol=1,angle_tol=15,primitive_cell=True)
    count = 0 ; count_u = 0 ; case1 = [] ; case2 = [] ; dic = {} ; n = 0
    for name2 in tqdm(glob.glob(cif_folder2+'/*.cif')):
    
        name = name2.split('\\')[-1].replace('_predicted.cif','_relaxed.cif')
        name1 = cif_folder1+'/'+name
        name1_unrelaxed = cif_folder1+'/'+name2.split('\\')[-1].replace('_predicted.cif','_unrelaxed.cif') ## origin
    
    #    name = name2.split('/')[-1]
    #    name1 = cif_folder1+'/'+name
    #    name1_unrelaxed = cif_folder1+'/'+name2.split('/')[-1].replace('_relaxed.cif','_unrelaxed.cif') ## origin
    
    #    name1 = cif_folder1+'/'+name2.split('/')[-1].replace('_unrelaxed.cif','_relaxed.cif') ## BOWSR
    #    name1_unrelaxed = cif_folder1+'/'+name2.split('/')[-1] #BOWSR
       
        atoms1 = read(name1)
        atoms2 = read(name2)
        atoms1_unrelaxed = read(name1_unrelaxed)
        print(atoms1, atoms2)
    
        struct1 = AseAtomsAdaptor.get_structure(atoms1)
        struct2 = AseAtomsAdaptor.get_structure(atoms2)
        struct1_unrelaxed = AseAtomsAdaptor.get_structure(atoms1_unrelaxed)
    
        d_u = sm2.get_rms_dist(struct1, struct1_unrelaxed)
        d = sm2.get_rms_dist(struct1, struct2)
        dic[name] = [d_u,d]
    
    #with open("diff_dict_1_1_15.json",'w') as f:
    #    json.dump(dic,f)
    
    #    if not sm.fit(struct1,struct1_unrelaxed):
    #        d_u = sm2.get_rms_dist(struct1, struct1_unrelaxed)
    #        d = sm2.get_rms_dist(struct1, struct2)
    #        count_u += 1
        n += 1
        if sm.fit(struct1,struct2):
            count +=1
    
    print(n,count,float(count/n))
    
    print(float(count_u)/float(len(glob.glob(cif_folder2+'/*.cif'))))
    
    with open("diff_augmentation_94_dict.json",'w') as f:
        json.dump(dic,f)

if "__main__" == __name__:
    main()
