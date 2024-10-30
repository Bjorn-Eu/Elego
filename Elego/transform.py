'''
Applies the 8 symmetry transformations of the plane
'''
import numpy as np

def id(np_array):
    return np_array

def rot90(np_array):
    return np.rot90(np_array,axes=(-2,-1))

def rot180(np_array):
    return np.rot90(np_array,2,axes=(-2,-1))

def rot270(np_array):
    return np.rot90(np_array,3,axes=(-2,-1))

def flip(np_array):
    return np.flip(np_array,axis=(-1))

def fliprt90(np_array):
    return rot90(flip(np_array))

def fliprt180(np_array):
    return rot180(flip(np_array))

def fliprt270(np_array):
    return rot270(flip(np_array))
