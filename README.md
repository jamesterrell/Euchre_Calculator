
# Euchre Calculator

An advanced Euchre game simulator that uses tree search algorithms to determine optimal play and calculate win probabilities. This tool helps Euchre players make strategic decisions by simulating all possible game outcomes assuming perfect play by all participants.

## Overview

Euchre is a trick-taking card game where strategy revolves around bidding (calling trump) and playing cards optimally. This simulator models the game using:

- **Vector-based card representation** for efficient computation
- **Tree search algorithms** to explore all possible plays
- **Optimal strategy simulation** assuming perfect play by all players
- **Probability calculations** with statistical confidence intervals

## Card Representation

Cards are represented as 2D vectors for computational efficiency:

```python
# Suit encoding:
# Hearts: negative x-axis [-14, 0] to [-9, 0] (Ace to 9)
# Diamonds: positive x-axis [9, 0] to [14, 0] (9 to Ace)
# Clubs: negative y-axis [0, -14] to [0, -9] (Ace to 9)
# Spades (Trump): positive y-axis [0, 90] to [0, 140] (9 to right bower)

# Example cards:
[-14, 0]   # Ace of hearts
[14, 0]    # Ace of diamonds
[0, -14]   # Ace of clubs
[0, 140]   # Right bower (Jack of spades - highest trump)
[0, 135]   # Left bower (Jack of clubs - second highest trump)
```

This representation naturally captures suit relationships and trump hierarchy, with spades always designated as trump.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/euchre-calculator.git
   cd euchre-calculator
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy numba jupyter
   ```

3. **Launch the interactive interface:**
   ```bash
   jupyter notebook interface.ipynb
   ```

## Quick Start

### Basic Simulation

```python
from deck import full_euchre_deck
from dealer import Dealer
from tree_search import definitive_winner
import numpy as np

# Create a dealer and deal random hands
dealer = Dealer(deck=full_euchre_deck, players=4)
dealer.deal_cards()
hands = np.array([dealer.hand1, dealer.hand2, dealer.hand3, dealer.hand4])

# Simulate optimal play assuming player 0 called trump
score = definitive_winner(
    dealt_hands=hands,
    starting_player=0,  # Player 0 leads first trick
    caller=0,          # Player 0 called trump
    verbose=True        # Show detailed play-by-play
)

print(f"Final score for calling team: {score}")
# Output: Final score for calling team: 2 (sweep), 1 (win), -2 (euchred)
```

### Probability Analysis with Stacked Hands

```python
from n_game_sim import generate_hands
from tree_search import definitive_winner

# Define a strong hand for analysis
strong_hand = np.array([
    [0, 140],  # Right bower
    [0, 135],  # Left bower
    [0, -9],   # 9 of clubs
    [-9, 0],   # 9 of hearts
    [9, 0]     # 9 of diamonds
])

upcard = np.array([[0, 90]])  # 9 of spades as upcard

# Generate 500 random games with your hand stacked
test_games = generate_hands(
    n_games=500,
    stack=strong_hand,
    stack_player=1,      # You are player 1
    up_card=upcard,
    up_card_player=2     # Upcard to player 2
)

# Run simulations
scores = np.zeros(500, dtype=np.int64)
for i in range(500):
    scores[i] = definitive_winner(
        dealt_hands=test_games[i],
        starting_player=3,  # Player 3 leads (after upcard pickup)
        caller=0,          # Player 0 called trump
        verbose=False
    )

# Calculate expected value and confidence interval
mean_score = np.mean(scores)
std_error = np.std(scores, ddof=1) / np.sqrt(len(scores))
ci_lower = mean_score - 1.96 * std_error
ci_upper = mean_score + 1.96 * std_error

print(f"Expected score: {mean_score:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

## Core Algorithm

The simulator uses a **tree search approach** to determine definitive winners:

1. **Opening Move Selection**: For each possible lead card, evaluate outcomes
2. **Response Optimization**: Each player responds optimally for their team
3. **Recursive Simulation**: Continue until all 5 tricks are played
4. **Score Calculation**: Determine final game outcome

### Key Functions

- **`definitive_winner()`**: Main simulation function returning final score
- **`find_best_opener()`**: Determines optimal opening card for current player
- **`find_best_response()`**: Simulates optimal responses from other players
- **`generate_hands()`**: Creates multiple random hand configurations

## Scoring System

- **+2**: Calling team takes all 5 tricks (march/sweep)
- **+1**: Calling team takes 3-4 tricks
- **-2**: Calling team takes 0-2 tricks (gets euchred)

## Project Structure

```
├── README.md                 # This file
├── deck.py                   # Card definitions and vector representations
├── dealer.py                 # Card dealing and hand management
├── tree_search.py            # Core tree search algorithms
├── n_play_round.py           # Single round simulation logic
├── n_branches.py             # Branch generation and filtering
├── n_game_sim.py             # Hand generation utilities
├── interface.ipynb           # Interactive Jupyter notebook
└── legacy_approach/          # Previous implementations
    ├── bit_string_approach.ipynb
    ├── branch_calc.py
    ├── play_round.py
    ├── sim_game_list_comp.py
    └── sim_game.py
```


## Applications

- **Learning Tool**: Understand optimal Euchre strategy
- **Decision Support**: Evaluate bidding decisions
- **Hand Analysis**: Assess strength of specific card combinations
- **Game Theory**: Study Nash equilibria in trick-taking games

## Dependencies

- **NumPy**: Numerical computing and array operations
- **Numba**: Just-in-time compilation for performance
- **Jupyter**: Interactive notebook environment (optional)

## License

MIT License - see LICENSE file for details


