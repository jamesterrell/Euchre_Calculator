import numpy as np
from numba import njit

@njit
def n_tricks(hands):
    tc = -1
    cards = int(hands.size/8)
    trees = np.zeros(shape=(cards**4, 4, 2), dtype=np.int64)
    for i in range(cards):
        for j in range(cards):
            for k in range(cards):
                for m in range(cards):
                    # Create a 2D numpy array explicitly
                    tc += 1
                    hand_combo = np.zeros((4, 2), dtype=np.int64)
                    hand_combo[0] = hands[0][i]
                    hand_combo[1] = hands[1][j]
                    hand_combo[2] = hands[2][k]
                    hand_combo[3] = hands[3][m]
                    trees[tc] = hand_combo
    return trees

@njit
def n_ap_filter(tricks, target, lead):
    mask = []
    for i in tricks:
        mask.append((i[lead] == target).all())
    return tricks[np.array(mask)]

@njit
def suit_id(arr):
    return np.arctan2(arr[1], arr[0])
    

@njit
def nfb_by_hand(branch, hand, target):
    x, y = target
    n = len(branch)
    mask = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if x == 0 and y > 0:
            mask[i] = branch[i][hand][0] == 0 and branch[i][hand][1] > 0
        elif x == 0 and y < 0:
            mask[i] = branch[i][hand][0] == 0 and branch[i][hand][1] < 0
        elif x > 0 and y == 0:
            mask[i] = branch[i][hand][0] > 0 and branch[i][hand][1] == 0
        elif x < 0 and y == 0:
            mask[i] = branch[i][hand][0] < 0 and branch[i][hand][1] == 0

    if len(branch[mask])==0:
        return branch

    return branch[mask]

@njit
def n_find_winner(trick, lead):
    # First, calculate norms for all cards
    norms = np.zeros(len(trick))
    for i, card in enumerate(trick):
        norms[i] = np.linalg.norm(card.astype(np.float64))
    
    # Check if any card has norm > 80
    if np.max(norms) > 80:
        return np.argmax(norms)
    else:
        # Find matching suits
        matching_suits = np.zeros(len(trick), dtype=np.bool_)
        for i, card in enumerate(trick):
            matching_suits[i] = (suit_id(card) == suit_id(trick[lead]))
        
        # If there are matching suits, find the winning card
        if np.any(matching_suits):
            # Calculate norms only for matching suits
            matching_norms = np.zeros(len(trick))
            for i in range(len(trick)):
                if matching_suits[i]:
                    matching_norms[i] = np.linalg.norm(trick[i].astype(np.float64))
                else:
                    matching_norms[i] = -1  # ensure non-matching suits don't win
            
            # Find the index of the winning card
            return np.argmax(matching_norms)
        
@njit
def n_winners(branch, lead):
    tricks = int(branch.size/8)
    score = np.zeros(tricks, dtype=np.int64)  # Change to 1D array
    for i in range(tricks):  # Iterate over indices
        winner = n_find_winner(trick=branch[i], lead=lead)
        score[i] = winner  # Direct indexing
    return score

@njit
def array_set_difference(arr1, arr2):
    # Create a result array with the same shape as input
    shape_arr = (arr1.shape[0],int(arr1.shape[1]-1),arr1.shape[2])
    result = np.zeros(shape=shape_arr, dtype=np.int64)
    
    # Iterate through each 2D subarray in arr1
    for i in range(arr1.shape[0]):
        # Track valid rows
        valid_rows = 0
        
        # Check each row in the current subarray
        for j in range(arr1.shape[1]):
            # Assume this row is valid until proven otherwise
            is_valid = True
            
            # Check against each row in arr2
            for k in range(arr2.shape[0]):
                if np.array_equal(arr1[i, j], arr2[k]):
                    is_valid = False
                    break
            
            # If row is valid, add it to result
            if is_valid:
                result[i, valid_rows] = arr1[i, j]
                valid_rows += 1
    
    return result
