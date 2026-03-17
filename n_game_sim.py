import numpy as np
from dealer import Dealer
from deck import full_euchre_deck

# might not need this anymore, but keeping it for now just in case.

def generate_hands(
    n_games=100,
    stack: np.array = None,
    stack_player: int = None,
    up_card: np.array = None,
    up_card_player: int = None,
):
    """Generate n random hand configurations."""
    all_hands = np.zeros((n_games, 4, 5, 2), dtype=np.int64)

    for i in range(n_games):
        game = Dealer(deck=full_euchre_deck, players=4)
        if stack is not None:
            game.stack_deck(stack_cards=stack, player=stack_player)
        if up_card is not None:
            game.stack_deck(stack_cards=up_card, player=up_card_player)
        game.deal_cards()
        all_hands[i] = np.array([game.hand0, game.hand1, game.hand2, game.hand3])

    return all_hands
