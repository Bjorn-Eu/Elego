import numpy as np

def rot90(np_array):
    return np.rot90(np_array,axes=(-2,-1))

def rot180(np_array):
    return np.rot90(np_array,2,axes=(-2,-1))

def rot270(np_array):
    return np.rot90(np_array,3,axes=(-2,-1))

def fliplr(np_array):
    return np.fliplr(np_array)

def fliprt90(np_array):
    return rot90(fliplr(np_array))

def fliprt180(np_array):
    return rot180(fliplr(np_array))

def fliprt270(np_array):
    return rot270(fliplr(np_array))
