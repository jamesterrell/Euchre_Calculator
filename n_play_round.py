from numba import njit
from n_branches import (
    n_tricks,
    n_ap_filter,
    nfb_by_hand,
)

@njit
def n_play_round(hands, lead, card_play):
    all_possible_tricks = n_tricks(hands)
    branch = n_ap_filter(tricks=all_possible_tricks, target=hands[lead][card_play], lead=lead)
    branch = nfb_by_hand(branch=branch, hand=(lead+1)%4, target=hands[lead][card_play])
    branch = nfb_by_hand(branch=branch, hand=(lead+2)%4, target=hands[lead][card_play])
    branch = nfb_by_hand(branch=branch, hand=(lead+3)%4, target=hands[lead][card_play])
    print(len(branch))
    return branch