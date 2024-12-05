import numpy as np

def calc_all_possible_hands(hands):
    grids = np.meshgrid(*[np.arange(len(a)) for a in hands], indexing='ij')
    product_indices = np.stack(grids, axis=-1).reshape(-1, len(hands))
    # Gather the cross product result based on indices, 625 results total
    all_possible_tricks = np.array([[hands[i][idx] for i, idx in enumerate(row)] for row in product_indices])
    return all_possible_tricks

def filter_branch_by_hand(branch, hand, column_idx, target):
    x, y = target
    ov_worst_eval = np.argmin([np.linalg.norm(i) for i in hand])
    ov_worst_card = hand[ov_worst_eval]

    if x == 0 and y > 0:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] > 0)]
    if x == 0 and y < 0:
        follows_suit = [coord for coord in hand if (coord[0] == 0 and coord[1] < 0)]
    if x > 0 and y == 0:
        follows_suit = [coord for coord in hand if (coord[1] == 0 and coord[0] > 0)]
    if x < 0 and y == 0:
        follows_suit = [coord for coord in hand if (coord[1] == 0 and coord[0] < 0)]

    follows_suit_arr = np.array(follows_suit)

    gt_eval = [np.linalg.norm(i) > np.linalg.norm(target) for i in follows_suit_arr]
    gt_arr = follows_suit_arr[gt_eval]

    trump_eval = [np.linalg.norm(i) > 80 for i in hand]
    trump_arr = hand[trump_eval]

    try:
        fs_worst_eval = np.argmin([np.linalg.norm(i) for i in follows_suit_arr])
        fs_worst_card = follows_suit_arr[fs_worst_eval]

    except ValueError:
        pass

    # if you can't follow suit but have trump, you'll either play trump or your worst card
    if len(follows_suit_arr) == 0 and len(trump_arr) > 0:
        matches = np.any(
            [
                np.all(branch[:, column_idx] == i, axis=1)
                for i in np.vstack([ov_worst_card, trump_arr])
            ],
            axis=0,
        )
        branch = branch[matches]

    # if you can follow suit, but can't beat the lead card, 
    # you'll play your worst card that follows suit
    elif len(follows_suit_arr) > 0 and len(gt_arr) == 0:
        matches = np.any(
            [np.all(branch[:, column_idx] == i, axis=1) for i in [fs_worst_card]],
            axis=0,
        )
        branch = branch[matches]

    # if you can't follow suit and have no trump, you'll play your worst card
    elif len(follows_suit_arr) == 0 and len(trump_arr) == 0:
        matches = np.any(
            [np.all(branch[:, column_idx] == i, axis=1) for i in [ov_worst_card]],
            axis=0,
        )
        branch = branch[matches]

    # for all other situations, we'll try all applicable cards 
    elif len(follows_suit_arr) > 0:
        matches = np.any(
            [np.all(branch[:, column_idx] == i, axis=1) for i in follows_suit_arr],
            axis=0,
        )
        branch = branch[matches]

    return branch


def compute_set_difference(arr1, arr2):
    results = []
    for a1_group, a2 in zip(arr1, arr2):
        group_result = []
        for a1 in a1_group:
            set_diff = np.array([row for row in a1 if row.tolist() not in a2.tolist()])
            group_result.append(set_diff)
        results.append(group_result)
    return results

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