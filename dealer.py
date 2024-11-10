import numpy as np
from dataclasses import dataclass

@dataclass
class dealer:
    deck: np.ndarray
    players: int
    
    def deal_cards(self) -> None:
        remaining_cards = np.arange(0, 24)
        
        # Create a dictionary to store hands
        self.hands = {}
        
        # Deal 5 cards to players
        for i in range(1, self.players+1):
            deal = np.random.choice(remaining_cards, size=5, replace=False)
            self.hands[f'hand{i}'] = self.deck[deal]
            remaining_cards = np.setdiff1d(remaining_cards, deal)
            
            #  set individual hand attributes
            setattr(self, f'hand{i}', self.deck[deal])