import os
import numpy as np


# 1 reticle
Cr = np.array([[0]])
Gr = np.array([[0,    0, 60000, 60000]])

current_script_path = os.path.abspath(__file__)
current_script_dir  = os.path.dirname(os.path.abspath(__file__))
npz_file = os.path.join(current_script_dir, f'./reticle_1r_{Gr[0][2]}_{Gr[0][3]}.npz')

print(current_script_path)
print(current_script_dir)
np.savez_compressed(npz_file, Cr=Cr, Gr=Gr)

npz_file = np.load(npz_file)
print(npz_file["Cr"]) # connectivity matrix of reticles
print(npz_file["Gr"]) # [x, y, w, h] of every reticles


# 2 reticle
Cr = np.array([[0, 1], 
               [1, 0]])

Gr = np.array([[0,     0, 25000, 14990], 
               [0, 15010, 25000, 14990]])


current_script_path = os.path.abspath(__file__)
current_script_dir  = os.path.dirname(os.path.abspath(__file__))
npz_file = os.path.join(current_script_dir, f'./reticle_2r_{Gr[0][2]}_{Gr[0][3]}.npz')

print(current_script_path)
print(current_script_dir)
np.savez_compressed(npz_file, Cr=Cr, Gr=Gr)

npz_file = np.load(npz_file)
print(npz_file["Cr"]) # connectivity matrix of reticles
print(npz_file["Gr"]) # [x, y, w, h] of every reticles


# 4 reticle
Cr = np.array([[0, 1, 0, 1], 
               [1, 0, 1, 0],
               [0, 1, 0, 1], 
               [1, 0, 1, 0]])

Gr = np.array([[    0,     0, 24990, 24990], 
               [25010,     0, 24990, 24990],
               [25010, 25010, 24990, 24990],
               [    0, 25010, 24990, 24990]])

current_script_path = os.path.abspath(__file__)
current_script_dir  = os.path.dirname(os.path.abspath(__file__))
npz_file = os.path.join(current_script_dir, f'./reticle_{Cr.shape[0]}r_{Gr[0][2]}_{Gr[0][3]}.npz')

print(current_script_path)
print(current_script_dir)
np.savez_compressed(npz_file, Cr=Cr, Gr=Gr)

npz_file = np.load(npz_file)
print(npz_file["Cr"]) # connectivity matrix of reticles
print(npz_file["Gr"]) # [x, y, w, h] of every reticles