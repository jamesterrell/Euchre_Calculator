
# Euchre Simulation

This project simulates a Euchre game to calculate the odds that a player's hand will result in a win, evaluating multiple rounds and factoring in trump cards, suit matching, and basic strategy. It can help Euchre players identify when to "order up" trump or when to pass. 

## Overview

The goal is to compute the probability of winning based on a player's hand. The simulation evaluates possible outcomes by simulating rounds, applying filtering techniques, and considering different strategies to predict win chances.

## Features

- **Card Representation:** Cards are represented as vectors with suits and values encoded in 2D space.
- **Trump Mechanics:** Spades are always considered trump.
- **Round Simulation:** Multiple rounds are simulated to assess different strategies.
- **Hand Evaluation:** The program filters possible actions based on the player's hand and the game state.
- **Meta-Simulation:** A broader simulation evaluates winning chances over many hands.

## Vector Representation of Euchre Deck
```python
import numpy as np

"""
Card suit representations:
- Hearts: negative x-axis coordinates
- Diamonds: positive x-axis coordinates
- Spades: positive y-axis coordinates
- Clubs: negative y-axis coordinates

For the purposes of this program, Spades is always considered trump, 
which is why it is assigned a higher value.
"""
full_euchre_deck = np.array(
    [
        [-9, 0],    # 9 of hearts
        [-10, 0],   # 10 of hearts
        [-11, 0],   # Jack of hearts
        [-12, 0],   # Queen of hearts
        [-13, 0],   # King of hearts
        [-14, 0],   # Ace of Hearts
        [0, -9],    # 9 of clubs
        [0, -10],   # 10 of clubs
        [0, -12],   # Queen of clubs
        [0, -13],   # King of clubs
        [0, -14],   # Ace of clubs (No Jack because it is the left bower)
        [9, 0],     # 9 of diamonds
        [10, 0],    # 10 of diamons
        [11, 0],    # Jack of diamonds
        [12, 0],    # Queen of diamonds
        [13, 0],    # King of diamonds
        [14, 0],    # Ace of diamonds
        [0, 90],    # 9 of spades (trump)
        [0, 100],   # 10 of spades (trump)
        [0, 110],   # Queen of spades (trump)
        [0, 120],   # King of spades (trump)
        [0, 130],   # Ace of spades (trump)
        [0, 135],   # left bower  (Jack of clubs)
        [0, 140],   # right bower (Jack of spades)
    ]
)
```
## File Structure

- `deck.py`: Contains the card definitions and the game logic related to the deck (e.g., defining suits and ranks).
- `n_play_round.py`: Implements the logic for simulating a single round of the game, including determining the optimal card play for each player based on their hand.
- `n_branches.py`: Defines functions for generating possible game branches and applying strategies like "trump or dump" and "smart loss."
- `meta_game.py`: Simulates a meta-game, where multiple hands are evaluated to determine the probability of a win for a given player.

## Key Functions

### `n_play_round(hands, lead, card_play)`
Simulates a round of Euchre based on the current hands and the card played by the lead player. It evaluates possible tricks, applies basic strategy functions (e.g., `common_sense`, `smart_loss`), and returns the possible outcomes (branches) for the round.

### `round1(hands_dealt)`
Simulates the first round of the game and calculates the possible outcomes for each card played by the lead player. It evaluates multiple branches, updating the hand configuration after each card is played.

### `next_round(current_hands, leads, game_round, game_score)`
Simulates the next round of the game, taking into account the current hands, leads, and the score. It calculates the possible outcomes for the round and updates the game state accordingly.

### `n_game_sim(game_hand, eval_position)`
Simulates the entire game based on a given array of four hands, evaluating the odds of winning for a specific player (identified by `eval_position`). It performs multiple rounds of simulation and computes the result based on the strategies and game rules.

### `meta_game_sim(meta_hands, eval_position)`
Simulates a meta-game by running multiple simulations of a given hand configuration. It evaluates the probability of winning for a specific player by averaging the results of the individual simulations.
### Dependencies
- `NumPy` for numerical operations.
- `Numba` for performance optimization (parallel processing).

## Usage

1. **Simulate a Game:** Use `n_game_sim` to simulate a single game and calculate the win probability for a given hand.
2. **Meta-Simulation:** Use `meta_game_sim` to simulate multiple hands and evaluate the average win probability.

### Example:

```python
from deck import full_euchre_deck
from dealer import Dealer
from n_game_sim import meta_game_sim
import numpy as np

evaluate = np.array([[0, 140], [0, 130], [0, 135], [0, -9], [9, 0]])
meta_hands = np.zeros(shape=(100, 4, 5, 2), dtype=np.int64)
for i in range(100):    
    game = Dealer(deck=full_euchre_deck ,players=4)
    game.stack_deck(stack_cards=evaluate, player=1)
    game.deal_cards()
    hands5 = np.array([game.hand1, game.hand2, game.hand3, game.hand4])
    meta_hands[i] = hands5

# a simple example that results in a 100% win rate
meta_game_sim(meta_hands=meta_hands, eval_position=0)

>>> 1.0
```

## License

MIT License


