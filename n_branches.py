import numpy as np
from numba import njit

@njit
def n_tricks(hands):
    tc = -1
    cards = int(hands.shape[1])
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
def suit_id(vector):
    # Manual magnitude calculation (equivalent to np.linalg.norm())
    magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
    
    # Cosine of angle is dot product divided by magnitudes
    cos_angle = float(vector[0]) / magnitude
    
    # Manual clip function
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    
    # Use arccos to get the angle
    angle_radians = np.arccos(cos_angle)
    
    # Adjust for vectors below x-axis
    if vector[1] < 0:
        angle_radians = 2 * np.pi - angle_radians

    return angle_radians
    

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

@njit
def common_sense(branch, target, player):
    store_cards = np.zeros(shape=(len(branch), 2), dtype=np.int64)
    for i in range(len(branch)):
        store_cards[i] = branch[i][player]

    # if the player can follow suit, see if they can beat the lead card
    store_bools = np.zeros(shape=(len(branch), 2), dtype=np.bool_)
    for j in range(len(store_cards)):
        store_bools[j] = (
            np.linalg.norm(store_cards[j].astype(np.float64))
            < np.linalg.norm(target.astype(np.float64))
        ) and (suit_id(store_cards[j]) == suit_id(target))

    # if that player can't beat the lead card, they'll play their worst card
    # that still follows suit
    if np.all(store_bools):
        store_vals = np.zeros(shape=(len(branch)), dtype=np.int64)
        for i in range(len(branch)):
            store_vals[i] = np.linalg.norm(branch[i][player].astype(np.float64))

        worst_card_ind = np.argmin(store_vals)

        common_sense_bools = np.zeros(shape=(len(branch)), dtype=np.bool_)
        for i in range(len(branch)):
            common_sense_bools[i] = np.all(
                branch[i][player] == store_cards[worst_card_ind]
            )

        return branch[common_sense_bools]

    else:
        return branch

@njit
def smart_loss(branch, target, player):
    num_branches = len(branch)
    store_cards = np.zeros((num_branches, 2), dtype=np.int64)
    
    # Explicitly copy cards for the given player
    for i in range(num_branches):
        store_cards[i, 0] = branch[i, player, 0]
        store_cards[i, 1] = branch[i, player, 1]

    # Prepare boolean masks
    store_bools = np.zeros(num_branches, dtype=np.int8)
    
    for j in range(num_branches):
        # Check that card is not trump
        card_norm = np.linalg.norm(store_cards[j].astype(np.float64))
        
        # Check suit conditions
        current_suit = suit_id(store_cards[j])
        target_suit = suit_id(target)
        
        # Set boolean condition
        store_bools[j] = (card_norm < 80.0) and (current_suit != target_suit)

    # If all cards can't follow suit and are not trump
    if np.all(store_bools):
        # Calculate card values
        store_vals = np.zeros(num_branches, dtype=np.float64)
        for i in range(num_branches):
            store_vals[i] = np.linalg.norm(branch[i, player].astype(np.float64))

        # Find worst card index
        worst_card_ind = np.argmin(store_vals)

        # Find branches with the worst card
        smart_loss_bools = np.zeros(num_branches, dtype=np.int8)
        worst_card_val = np.linalg.norm(store_cards[worst_card_ind].astype(np.float64))
        
        for i in range(num_branches):
            branch_val = np.linalg.norm(store_cards[i].astype(np.float64))
            smart_loss_bools[i] = np.isclose(branch_val, worst_card_val)

        # Return filtered branches
        return branch[smart_loss_bools == 1]

    else:
        return branch
    

@njit
def trump_or_dump(branch, target, player):
    num_branches = branch.shape[0]
    store_cards = np.zeros((num_branches, 2), dtype=branch.dtype)
    
    # Explicitly copy cards for the given player
    for i in range(num_branches):
        store_cards[i, 0] = branch[i, player, 0]
        store_cards[i, 1] = branch[i, player, 1]

    # Prepare boolean masks
    store_bools = np.zeros(num_branches, dtype=np.bool_)
    
    for j in range(num_branches):
        # Check that card is not trump - convert to float for norm
        card_norm = np.linalg.norm(store_cards[j].astype(np.float64))
        
        # Check suit conditions - convert to float for suit calculation
        current_suit = suit_id(store_cards[j])
        target_suit = suit_id(target)
        
        # Set boolean condition
        store_bools[j] = (card_norm > 80.0) or (current_suit != target_suit)

    # Player has trump but can not follow suit
    if np.all(store_bools):
        # Calculate card values
        store_vals = np.zeros(num_branches, dtype=branch.dtype)
        for i in range(num_branches):
            store_vals[i] = np.linalg.norm(branch[i, player].astype(np.float64))

        # Find worst card index
        worst_card_ind = np.argmin(store_vals)

        # find trump cards
        trump_cards = store_vals > 80

        # find worst card value
        worst_card_val = np.linalg.norm(store_cards[worst_card_ind].astype(np.float64))

        # Find worst cards
        worst_cards = store_vals == worst_card_val

        # Find branches with the worst card or trump card
        trump_or_dump_bools = np.zeros(num_branches, dtype=np.bool_)

        for i in range(num_branches):
            trump_or_dump_bools[i] = trump_cards[i] or worst_cards[i]

        # Return filtered branches
        return branch[trump_or_dump_bools]

    else:
        return branch