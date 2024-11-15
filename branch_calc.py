import numpy as np

def calc_all_possible_hands(hands):
    grids = np.meshgrid(*[np.arange(len(a)) for a in hands], indexing='ij')
    product_indices = np.stack(grids, axis=-1).reshape(-1, len(hands))
    # Gather the cross product result based on indices, 625 results total
    all_possible_tricks = np.array([[hands[i][idx] for i, idx in enumerate(row)] for row in product_indices])
    return all_possible_tricks

def filter_branch_by_hand(branch, hand, column_idx, target):
    x, y = target
    if x==0 and y>0:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] > 0) or coord[1] == -135]
    if x==0 and y<0:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] < 0 and coord[1]>-15)]
    if x==0 and y==-135:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] > 0)]
    if x>0 and y==0: 
        follows_suit = [coord for coord in hand if (coord[1] == 0 and coord[0] > 0)]
    if x<0 and y==0: 
        follows_suit = [coord for coord in hand if (coord[1] == 0 and coord[0] < 0)]

    follows_suit_arr = np.array(follows_suit)
    
    if len(follows_suit_arr) > 0:
        matches = np.any([np.all(branch[:, column_idx] == i, axis=1) for i in follows_suit_arr], axis=0)
        branch = branch[matches]
    
    return branch


def setdiff2d_idx(arr1, arr2):
    delta = set(map(tuple, arr2))
    idx = [tuple(x) not in delta for x in arr1]
    return np.array(arr1[idx])

# use the angle with respect to the x-axis to indetify suit for a card
def suit_id(arr):
    return np.arctan2(arr[1], arr[0])

def find_winner(lead, trick):
    if np.max([np.linalg.norm(card) for card in trick])>80:
        return np.argmax([np.linalg.norm(card) for card in trick])
    else:
        matching_suits = [card for card in trick if suit_id(card) == suit_id(lead)]
        winning_card = max(matching_suits, key=np.linalg.norm)
        winning_card_index = np.where((trick == winning_card).all(axis=1))[0][0]
        return winning_card_index