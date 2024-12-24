import numpy as np
from numba import njit


@njit
def n_tricks(hands):
    """
    Generates all possible combinations of cards played by four players in a euchre trick.

    Args:
        hands (np.ndarray): A 3D array of shape (4, n, 2), where:
            - The first dimension represents the four players.
            - The second dimension represents the number of cards each player holds.
            - The third dimension (size 2) represents the suit and value of a card.

    Returns:
        np.ndarray: A 3D array of shape (n^4, 4, 2), where:
            - `n^4` represents all possible combinations of cards played by the four players.
            - The second dimension represents the four players.
            - The third dimension (size 2) represents the suit and value of each card in the combination.

    Notes:
        - The resulting `trees` array contains every possible combination of one card from each player,
          maintaining the order of players.
    """
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
    """
    Filters a set of euchre tricks the ahere to suit following rules based on the first card played (target)
    and the player that played it (lead)

    Args:
        tricks (np.ndarray): A 3D array of shape (m, 4, 2), where:
            - `m` is the number of tricks.
            - The second dimension represents the four players.
            - The third dimension (size 2) represents the suit and value of each card in the trick.
        target (np.ndarray): A 1D array of size 2 representing the target card (suit and value).
        lead (int): The index of the lead player (0 through 3) whose card is checked against the target.

    Returns:
        np.ndarray: A filtered 3D array of tricks that meet the condition, retaining the original shape structure
        but with fewer entries if some tricks are excluded.
    """
    mask = []
    for i in tricks:
        mask.append((i[lead] == target).all())
    return tricks[np.array(mask)]


@njit
def suit_id(vector):
    """
    Computes the angle of a 2D vector in radians, relative to the positive x-axis,
    using its magnitude and direction. This is used to identify the suit of the card.

    Args:
        vector (np.ndarray): A 1D array of size 2 representing the x and y components of the vector.

    Returns:
        float: The angle of the vector in radians, ranging from 0 to 2π, measured counterclockwise
        from the positive x-axis.
    """
    # Manual magnitude calculation (equivalent to np.linalg.norm())
    magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)

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
    """
    Filters a set of card branches by matching conditions for a specific hand and target vector.

    Args:
        branch (np.ndarray): A 3D array of shape (m, 4, 2), where:
            - `m` is the number of branches (sets of potential card combinations).
            - The second dimension represents the four players (hands).
            - The third dimension (size 2) represents the x and y components of the card's vector.
        hand (int): The index of the player's hand to evaluate (0 through 3).
        target (tuple): A 2D vector `(x, y)` specifying the conditions to filter by:
            - `(0, y > 0)` filters for cards with `x == 0` and `y > 0`.
            - `(0, y < 0)` filters for cards with `x == 0` and `y < 0`.
            - `(x > 0, 0)` filters for cards with `x > 0` and `y == 0`.
            - `(x < 0, 0)` filters for cards with `x < 0` and `y == 0`.

    Returns:
        np.ndarray: A filtered 3D array of branches that meet the target conditions.
        If no branches match, the original `branch` array is returned unchanged.

     Notes:
        - The function applies conditions based on the vector representation of cards in a given hand.
        - The mask ensures that only branches satisfying the `target` criteria are retained.
    """

    # realistically I could refactor this to use suit_id. This just works and is easy to read.
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

    if len(branch[mask]) == 0:
        return branch

    return branch[mask]


@njit
def n_find_winner(trick, lead):
    """
    Determines the winner of a euchre trick based on card norms and suit matching.

    Args:
        trick (np.ndarray): A 2D array of shape (4, 2), where:
            - Each row represents a card played in the trick, expressed as a vector (x, y).
            - The cards are indexed in the order they were played.
        lead (int): The index of the lead player (0 through 3) who initiated the trick.

    Returns:
        int: The index of the player who won the trick.

    Logic:
        1. Calculates the norm (magnitude) of each card vector.
        2. If one or more cards have a norm greater than 80, the card with the highest norm > 80 wins the trick.
        3. If no cards have a norm > 80:
            - Identifies cards with the same suit as the lead card.
            - Among matching suits, determines the card with the highest norm as the winner.
            - If no cards match the lead suit, the trick's winner defaults to the player
              with the highest card norm overall.

    Notes:
        - The `suit_id` function is used to determine a card's suit based on its vector.
        - The norm of a card is treated as its "strength" within a given suit.
        - Cards with norms less than 80 are only compared if no dominant card (>80 norm) is present.
    """
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
            matching_suits[i] = suit_id(card) == suit_id(trick[lead])

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
def array_set_difference(arr1, arr2):
    """
    Computes the set difference between rows of 2D subarrays in `arr1` and rows of `arr2`.

    Args:
        arr1 (np.ndarray): A 3D array of shape `(m, n, p)`, where:
            - `m` is the number of subarrays in `arr1`.
            - `n` is the number of rows (vectors) in each subarray.
            - `p` is the dimensionality of each vector.
        arr2 (np.ndarray): A 2D array of shape `(q, p)`, where:
            - `q` is the number of rows (vectors) in `arr2`.
            - `p` is the dimensionality of each vector.

    Returns:
        np.ndarray: A 3D array of shape `(m, n-1, p)` containing rows from each subarray of `arr1`
        that are not present in `arr2`. Rows are preserved in the order they appear in `arr1`.

    Logic:
        - Iterates through each 2D subarray in `arr1`.
        - For each subarray, iterates through its rows and checks whether each row is present in `arr2`.
        - If a row is not present in `arr2`, it is added to the result array for that subarray.
        - The result array omits rows that are found in `arr2`.

    Notes:
        - The function assumes `arr1` and `arr2` contain integer arrays (dtype `np.int64`).
        - The result array has a reduced number of rows (`n-1`) compared to `arr1`, as it accounts for rows that are removed.
    """
    # Create a result array with the same shape as input
    shape_arr = (arr1.shape[0], int(arr1.shape[1] - 1), arr1.shape[2])
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
    """
    Implements a strategy for a player in a trick, determining which card to play
    based on whether they can beat the lead card while following the suit.

    Args:
        branch (np.ndarray): A 3D array of shape `(n, m, p)` representing the set of
            possible card plays, where:
            - `n` is the number of possible plays in the trick.
            - `m` is the number of players (typically 4).
            - `p` is the dimensionality of each card vector (representing the card as a vector).
        target (np.ndarray): A 2D array of shape `(1, 2)` representing the lead card played
            in the trick.
        player (int): The index of the player (0 through 3) whose strategy is being evaluated.

    Returns:
        np.ndarray: A filtered subset of `branch` representing the possible plays for the player,
        with the following logic:
            - If the player cannot beat the lead card but can follow suit, they will play their weakest card
              that still follows the suit.

    Notes:
        - The `suit_id` function is used to identify the suit of the cards.
        - The function compares the strength (norm) of the cards to determine 
          which one can or cannot beat the lead card.
    """
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
    """
    Implements a strategy for a player when they cannot follow suit, deciding
    which card to play to minimize the loss by playing the weakest card.

    Args:
        branch (np.ndarray): A 3D array of shape `(n, m, p)` representing the set of
            possible card plays, where:
            - `n` is the number of possible plays in the trick.
            - `m` is the number of players (typically 4).
            - `p` is the dimensionality of each card vector (representing the card as a vector).
        target (np.ndarray): A 2D array of shape `(1, 2)` representing the lead card
            played in the trick.
        player (int): The index of the player (0 through 3) whose strategy is being evaluated.

    Returns:
        np.ndarray: A filtered subset of `branch` representing the possible plays for the player,
        with the following logic:
            - If the player cannot follow suit and their cards are not trump cards,
              they will play the weakest card to minimize the loss.
            - If the player has cards that follow suit or are trump, the original `branch` is returned.

    Notes:
        - The `suit_id` function is used to identify the suit of the cards.
        - The norm of the card is used to determine card strength.
    """
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
    """
    Implements a strategy for a player when they have trump cards and cannot follow suit.
    In this case, the player will either play trump or their worst card.

    Args:
        branch (np.ndarray): A 3D array of shape `(n, m, p)` representing the set of
            possible card plays, where:
            - `n` is the number of possible plays in the trick.
            - `m` is the number of players (typically 4).
            - `p` is the dimensionality of each card vector (representing the card as a vector).
        target (np.ndarray): A 2D array of shape `(1, 2)` representing the lead card
            played in the trick.
        player (int): The index of the player (0 through 3) whose strategy is being evaluated.

    Returns:
        np.ndarray: A filtered subset of `branch` representing the possible plays for the player,
        with the following logic:
            - If the player cannot follow suit and has trump cards, they will either  play a trump card
            or dump the worst card in their hand.

    Notes:
        - The `suit_id` function is used to identify the suit of the cards.
        - The norm of the card is used to determine card strength and whether the card is a trump card.
        - Cards with a norm greater than 80 are considered trump cards.
    """
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
