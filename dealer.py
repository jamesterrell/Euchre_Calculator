import numpy as np
from dataclasses import dataclass

HAND_SIZE = 5


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
        if self.players < 1:
            raise ValueError("players must be at least 1, got %r" % (self.players,))
        if self.players * HAND_SIZE > len(self.deck):
            raise ValueError(
                "deck of %d cards cannot deal %d cards to %d players"
                % (len(self.deck), HAND_SIZE, self.players)
            )
        self.hands = {}
        for i in range(self.players):
            self.hands[f"hand{i}"] = []

    def stack_deck(self, stack_cards: np.array, player: int):
        """
        Stacks specific cards into a player's hand and removes them from the deck.

        Cards accumulate: calling this twice for the same player adds to that
        player's hand rather than replacing what is already there.

        Args:
            stack_cards (np.array): The card(s) to assign to the player's hand,
                shaped (n, 2), or (2,) for a single card.
            player (int): The player index to whom the cards are assigned.

        Returns:
            np.array: The updated hand for the specified player.

        Raises:
            ValueError: If `player` is not a seat at this table, if the cards are
                not shaped like cards, if stacking them would overfill the hand, or
                if a card in `stack_cards` is not present in the deck -- which means
                it was already dealt or stacked elsewhere.
        """
        if not 0 <= player < self.players:
            raise ValueError(
                "player must be in 0..%d, got %r" % (self.players - 1, player)
            )

        stack_cards = np.asarray(stack_cards)
        # accept a bare card as well as an array of them
        if stack_cards.ndim == 1:
            stack_cards = stack_cards[None, :]
        if stack_cards.ndim != 2 or stack_cards.shape[1] != 2:
            raise ValueError(
                "stack_cards must be shaped (n, 2) or (2,), got %r"
                % (stack_cards.shape,)
            )

        key = f"hand{player}"
        held = len(self.hands[key])
        if held + len(stack_cards) > HAND_SIZE:
            raise ValueError(
                "stacking %d cards onto player %d's %d would exceed the %d-card hand"
                % (len(stack_cards), player, held, HAND_SIZE)
            )

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

        # Append rather than overwrite. Overwriting dropped the earlier cards from
        # the hand while leaving them removed from the deck, so they could not be
        # dealt to anyone -- generate_hands hit this whenever `stack` and `up_card`
        # named the same player.
        if held:
            self.hands[key] = np.vstack((self.hands[key], stack_cards))
        else:
            self.hands[key] = stack_cards
        return self.hands[key]

    def deal_cards(self) -> None:
        """
        Deals cards to all players, ensuring each player has exactly 5 cards in their hand.
        Cards are randomly selected from the remaining deck.
        """
        remaining_cards = np.arange(0, len(self.deck))

        # Deal 5 cards to players
        for i in self.hands:
            card_count = len(self.hands[i])
            if card_count == HAND_SIZE:
                setattr(self, i, self.hands[i])
                continue

            needed = HAND_SIZE - card_count
            if needed > len(remaining_cards):
                raise ValueError(
                    "deck exhausted: %s needs %d more cards but %d remain"
                    % (i, needed, len(remaining_cards))
                )
            deal = np.random.choice(remaining_cards, size=needed, replace=False)
            if card_count == 0:
                dealt_hand = self.deck[deal]
            else:
                dealt_hand = np.vstack((self.hands[i], self.deck[deal]))
            self.hands[i] = dealt_hand
            remaining_cards = np.setdiff1d(remaining_cards, deal)

            #  set individual hand attributes
            setattr(self, i, dealt_hand)
