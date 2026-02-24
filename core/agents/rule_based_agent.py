import numpy as np
import torch

class RuleBasedAgent:
    def __init__(self, style="aggressive", device="cpu"):
        """
        style: 
            - "aggressive": Raise > Check > Call > Fold
            - "conservative": Check > Fold > Call > Raise
            - "random": Random valid action
        """
        self.style = style
        self.device = device
        
        # Leduc Hold'em action mapping
        self.ACTION_CALL = 0
        self.ACTION_RAISE = 1
        self.ACTION_FOLD = 2
        self.ACTION_CHECK = 3

    def get_action(self, obs, mask, deterministic=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (action_tensor, log_prob_tensor)
        """
        # make sure obs and mask are numpy arrays
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().flatten()
        else:
            mask_np = np.array(mask).flatten()
            
        valid_actions = np.where(mask_np == 1)[0]
        
        if len(valid_actions) == 0:
            chosen_action = 0  # default to action 0 if no valid actions (should not happen in a well-formed environment)
            
        elif self.style == "aggressive":
            if self.ACTION_RAISE in valid_actions:
                chosen_action = self.ACTION_RAISE
            elif self.ACTION_CHECK in valid_actions:
                chosen_action = self.ACTION_CHECK
            elif self.ACTION_CALL in valid_actions:
                chosen_action = self.ACTION_CALL
            else:
                chosen_action = valid_actions[0]
                
        elif self.style == "conservative":
            if self.ACTION_CHECK in valid_actions:
                chosen_action = self.ACTION_CHECK
            elif self.ACTION_FOLD in valid_actions:
                chosen_action = self.ACTION_FOLD
            elif self.ACTION_CALL in valid_actions:
                chosen_action = self.ACTION_CALL
            else:
                chosen_action = valid_actions[0]
                
        else:  # random
            chosen_action = np.random.choice(valid_actions)

        action_tensor = torch.tensor([chosen_action], dtype=torch.long, device=self.device)
        # Since this is a deterministic rule-based agent, we can set log_prob to 0 for the chosen action
        log_prob_tensor = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        
        return action_tensor, log_prob_tensor