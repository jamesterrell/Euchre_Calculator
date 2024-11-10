import numpy as np

def filter_branch_by_hand(branch, hand, column_idx, target):
    x, y = target
    if x==0 and y>0:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] > 0) or coord[1] == -14]
    if x==0 and y<0:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] < 0) or coord[1] == 14]
    if x>0 and y==0: 
        follows_suit = [coord for coord in hand if (coord[1] == 0 and coord[0] > 0) or coord[0] == -14]
    if x<0 and y==0: 
        follows_suit = [coord for coord in hand if (coord[1] == 0 and coord[0] < 0) or coord[0] == 14]

    follows_suit_arr = np.array(follows_suit)
    
    if len(follows_suit_arr) > 0:
        matches = np.any([np.all(branch[:, column_idx] == i, axis=1) for i in follows_suit_arr], axis=0)
        branch = branch[matches]
    
    return branch