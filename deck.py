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
