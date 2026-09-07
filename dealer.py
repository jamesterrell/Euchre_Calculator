import numpy as np
from dataclasses import dataclass


@dataclass
class Dealer:
    """
    Represents a euchre dealer responsible for shuffling, dealing, and managing the deck
    and player hands during the game.

    Attributes:
        deck (np.ndarray): The array representing the deck of cards, where each card is a unique representation.
        players (int): The number of players in the game.
    """

    deck: np.ndarray
    players: int

    def __post_init__(self):
        """
        Initializes the dealer with empty hands for each player after the class is instantiated.
        """
        self.hands = {}
        for i in range(4):
            self.hands[f"hand{i}"] = []

    def stack_deck(self, stack_cards: np.array, player: int):
        """
        Stacks specific cards into a player's hand and removes them from the deck.

        Args:
            stack_cards (np.array): The array of cards to assign to the player's hand.
            player (int): The player index (0 through 3) to whom the cards are assigned.

        Returns:
            np.array: The updated hand for the specified player.

        Raises:
            ValueError: If a card in `stack_cards` is not present in the deck,
                which means it was already dealt or stacked elsewhere.
        """
        self.hands[f"hand{player}"] = stack_cards
        # Match whole cards. np.isin here would compare each coordinate against
        # every value in stack_cards, so a card was removed whenever both of its
        # numbers happened to appear anywhere in the stack -- e.g. stacking
        # 9d [9, 0] and Ac [0, -14] also removed Ah [-14, 0].
        remove = (self.deck[:, None, :] == stack_cards[None, :, :]).all(axis=2).any(axis=1)
        if remove.sum() != len(stack_cards):
            raise ValueError(
                f"stack_deck: matched {remove.sum()} of {len(stack_cards)} cards in the deck"
            )
        self.deck = self.deck[~remove]
        return self.hands[f"hand{player}"]

    def deal_cards(self) -> None:
        """
        Deals cards to all players, ensuring each player has exactly 5 cards in their hand.
        Cards are randomly selected from the remaining deck.
        """
        remaining_cards = np.arange(0, len(self.deck))

        # Deal 5 cards to players
        for i in self.hands:
            card_count = len(self.hands[i])
            if card_count == 0:
                deal = np.random.choice(remaining_cards, size=5, replace=False)
                dealt_hand = self.deck[deal]
                self.hands[i] = dealt_hand
                remaining_cards = np.setdiff1d(remaining_cards, deal)
            elif card_count == 5:
                setattr(self, i, self.hands[i])
                continue
            else:
                deal = np.random.choice(
                    remaining_cards, size=5 - card_count, replace=False
                )
                # print(self.hands[i])
                dealt_hand = np.vstack((self.hands[i], self.deck[deal]))
                self.hands[i] = dealt_hand
                remaining_cards = np.setdiff1d(remaining_cards, deal)

            #  set individual hand attributes
            setattr(self, i, dealt_hand)
