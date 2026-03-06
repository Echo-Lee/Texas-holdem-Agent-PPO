# play_game.py
import os
import yaml
import numpy as np
import gradio as gr
import torch
from pettingzoo.classic import texas_holdem_v4
from pathlib import Path
# Compatibility layer using PPO nets directly (no RLAgent wrapper)

# -----------------------------
# Config
# -----------------------------
ACTIONS = {0: "Call", 1: "Raise", 2: "Fold", 3: "Check"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARD_DIR = r"D:\RL Project\poker_cards"  # Update this path to your local card images folder

BACK_CODE = "BACK"
BACK_PATH = os.path.join(CARD_DIR, f"{BACK_CODE}.png")

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]

DEFAULT_INITIAL_STACK = 100  # used only when starting a brand-new match


# -----------------------------
# Observation decoding helpers
# -----------------------------
def decode_cards(obs72: np.ndarray):
    cards = []
    bits = obs72[:52].astype(int)
    for suit_idx, suit in enumerate(SUITS):
        base = 13 * suit_idx
        for r in range(13):
            if bits[base + r] == 1:
                cards.append(f"{RANKS[r]}{suit}")
    return cards


def decode_round_raises(obs72: np.ndarray):
    raises = []
    offset = 52
    for rnd in range(4):
        onehot = obs72[offset + 5 * rnd : offset + 5 * (rnd + 1)]
        raises.append(int(np.argmax(onehot)) if onehot.sum() else None)
    return raises


def card_code_to_path(code: str):
    if code == BACK_CODE:
        return BACK_PATH if os.path.exists(BACK_PATH) else None
    p = os.path.join(CARD_DIR, f"{code}.png")
    return p if os.path.exists(p) else None


def pad_with_back(cards, total):
    cards = list(cards)
    if len(cards) < total:
        cards += [BACK_CODE] * (total - len(cards))
    return cards[:total]


def cards_to_gallery_items(cards):
    items = []
    for c in cards:
        p = card_code_to_path(c)
        if p is not None:
            cap = "" if c == BACK_CODE else c
            items.append((p, cap))
    return items


def split_board_and_holes_preserve_order(cards_p0, cards_p1):
    set1 = set(cards_p1)
    board = [c for c in cards_p0 if c in set1]
    board_set = set(board)
    hole0 = [c for c in cards_p0 if c not in board_set]
    hole1 = [c for c in cards_p1 if c not in board_set]
    return board, hole0, hole1


def render_state_md(obs_h, legal_mask, done, winner_text, stacks_text):
    raises = decode_round_raises(obs_h["observation"])
    raises_str = ", ".join([str(x) if x is not None else "-" for x in raises])

    # legal = [ACTIONS[i] for i in range(4) if legal_mask[i] == 1]
    # legal_str = ", ".join(legal) if legal else "(not your turn)"

    end_line = f"\n\n## {winner_text}\n" if done else ""
    return f"""
{stacks_text}
**Round raises buckets (0-4):** `{raises_str}`  
{end_line}
"""



# -----------------------------
# Agent wrapper
# -----------------------------
class AgentPolicy:
    def __init__(self, model_path: str, config_path: str = "running_config.yaml", device: str = "cpu"):
        self.device = torch.device(device)
        # Minimal compatibility layer: hard-code observation dimension for Texas Hold'em
        obs_dim = 72  # observation vector length used by existing code
        obs_dim += 4   # add one-hot encoding of opponent's last action (for better opponent modeling)
        from core.networks.policy_value_network import PolicyNet
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        agent_cfg = cfg.get('model', {}).get('agent', {'hidden_layers': [256, 256], 'use_layer_norm': True})
        self.policy_net = PolicyNet(
            input_dim=obs_dim, 
            output_dim=4, 
            **agent_cfg
        ).to(self.device)
        self.load(model_path)

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "policy_state_dict" in state:
            state = state["policy_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.policy_net.load_state_dict(state, strict=False)
            print(f"[OK] Loaded model with strict=False: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")

    @torch.no_grad()
    def act(self, obs_dict):
        obs = obs_dict["observation"].astype(np.float32) # 这是 72 维
        mask = obs_dict["action_mask"].astype(np.int64)

        legal = np.where(mask == 1)[0]
        if legal.size == 0:
            return 3

        x_raw = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        
        padding = torch.zeros((1, 4), device=self.device)
        x = torch.cat([x_raw, padding], dim=-1)
        illegal_mask = (torch.from_numpy(mask).to(self.device) == 0).unsqueeze(0)

        logits = self.policy_net(x)

        masked_logits = logits.clone()
        masked_logits.masked_fill_(illegal_mask, -1e9)
        
        action = torch.argmax(masked_logits, dim=-1).item()
        return int(action)


# -----------------------------
# One-hand session (no bankroll inside)
# -----------------------------
class PokerHand:
    def __init__(self, agent_policy: AgentPolicy, seed=None):
        self.env = texas_holdem_v4.env(num_players=2)
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        self.seed = int(seed)

        self.env.reset(seed=self.seed)
        self.human = "player_0"
        self.bot = "player_1"
        self.agent_policy = agent_policy

        self.done = False
        self.action_log = []  # actions within this hand

        self.action_log.append(f"=== Hand (seed={self.seed}) ===")
        self._auto_play_until_human()

    def _obs(self, agent):
        obs = self.env.observe(agent)
        if obs is None:
            return {"observation": np.zeros(72, dtype=np.float32), "action_mask": np.ones(4, dtype=np.int64)}
        return obs

    def _current_agent(self):
        return self.env.agent_selection

    def _is_terminal(self):
        return all(self.env.terminations.values()) or all(self.env.truncations.values())

    def _step_checked(self, action_int):
        self.env.step(int(action_int))
        self.done = self._is_terminal()

    def _auto_play_until_human(self):
        while not self.done:
            agent = self._current_agent()

            if self.env.terminations[agent] or self.env.truncations[agent]:
                self.env.step(None)
                self.done = self._is_terminal()
                continue

            if agent == self.bot:
                obs = self._obs(self.bot)
                mask = obs["action_mask"].astype(int)

                a = self.agent_policy.act(obs)
                if mask[a] != 1:
                    legal = np.where(mask == 1)[0]
                    a = int(legal[0]) if legal.size else 3

                self.action_log.append(f"Agent ({self.bot}): {ACTIONS[a]}")
                self._step_checked(a)
                continue

            if agent == self.human:
                break

            self.env.step(None)

    def step_human(self, action_int):
        if self.done:
            self.action_log.append("Hand is over.")
            return

        if self._current_agent() != self.human:
            self._auto_play_until_human()

        if self.done:
            return

        obs = self._obs(self.human)
        mask = obs["action_mask"].astype(int)

        if mask[action_int] != 1:
            self.action_log.append(f"Human ({self.human}): ILLEGAL {ACTIONS[action_int]} (mask={mask.tolist()})")
            self._step_checked(action_int)
            self._auto_play_until_human()
        else:
            self.action_log.append(f"Human ({self.human}): {ACTIONS[action_int]}")
            self._step_checked(action_int)
            self._auto_play_until_human()

    def rewards(self):
        return float(self.env.rewards.get(self.human, 0.0)), float(self.env.rewards.get(self.bot, 0.0))

    def winner_text(self):
        r0, r1 = self.rewards()
        eps = 1e-9
        if r0 > r1 + eps:
            return f"You win! (reward {r0:.3f} vs {r1:.3f})"
        elif r1 > r0 + eps:
            return f"Agent wins. (reward {r0:.3f} vs {r1:.3f})"
        return f"Tie. (reward {r0:.3f} vs {r1:.3f})"

    def galleries(self):
        obs0 = self._obs(self.human)
        obs1 = self._obs(self.bot)

        cards0 = decode_cards(obs0["observation"])
        cards1 = decode_cards(obs1["observation"])

        board, hole0, hole1 = split_board_and_holes_preserve_order(cards0, cards1)

        board_display = pad_with_back(board, 5)
        hero_display = pad_with_back(hole0, 2)

        opp_display = hole1 if self.done else [BACK_CODE, BACK_CODE]
        opp_display = pad_with_back(opp_display, 2)

        return (
            cards_to_gallery_items(board_display),
            cards_to_gallery_items(hero_display),
            cards_to_gallery_items(opp_display),
            obs0["action_mask"].astype(int),
            obs0,
        )


# -----------------------------
# Match state (persists across hands)
# -----------------------------
class MatchState:
    def __init__(self, agent_policy: AgentPolicy, initial_stack=DEFAULT_INITIAL_STACK):
        self.agent_policy = agent_policy
        self.initial_stack = float(initial_stack)
        self.stack = {"player_0": float(initial_stack), "player_1": float(initial_stack)}

        self.hand_no = 0
        self.history = []
        self.hand = None
        self.payout_applied = False

        # cursor: how many lines of current hand.action_log we have copied into history
        self.hand_log_pos = 0

    def stacks_text(self):
        return (
            f"**Chips (start={int(self.initial_stack)}):**  "
            f"player_0 = `{self.stack['player_0']:.3f}` | player_1 = `{self.stack['player_1']:.3f}`"
        )

    def start_new_hand(self, seed=None):
        self.apply_payout_if_done()

        self.hand_no += 1
        self.payout_applied = False

        self.history.append("\n==============================")
        self.history.append(f"Hand {self.hand_no} start")

        self.hand = PokerHand(self.agent_policy, seed=seed)

        # Copy ALL lines currently produced (may include multiple Agent actions)
        self.history.extend(self.hand.action_log)
        self.hand_log_pos = len(self.hand.action_log)

    def sync_new_hand_log_lines(self):
        """
        Append any NEW lines from hand.action_log into history.
        This is the key fix: do not append just one line.
        """
        if self.hand is None:
            return
        new_lines = self.hand.action_log[self.hand_log_pos :]
        if new_lines:
            self.history.extend(new_lines)
            self.hand_log_pos = len(self.hand.action_log)

    def apply_payout_if_done(self):
        if self.hand is None or not self.hand.done:
            return
        if self.payout_applied:
            return
        r0, r1 = self.hand.rewards()
        self.stack["player_0"] += r0
        self.stack["player_1"] += r1
        self.history.append(f"[PAYOUT] player_0: {r0:+.3f}, player_1: {r1:+.3f}")
        self.history.append(f"[STACKS] player_0: {self.stack['player_0']:.3f}, player_1: {self.stack['player_1']:.3f}")
        self.payout_applied = True

    def view_outputs(self):
        if self.hand is None:
            return f"{self.stacks_text()}\n\nClick **New Hand** to start.", "Ready.", [], [], [], "\n".join(self.history)

        board_g, hero_g, opp_g, legal_mask, obs0 = self.hand.galleries()
        
        legal = [ACTIONS[i] for i in range(4) if legal_mask[i] == 1]
        legal_hint = f"✅ **Legal Actions:** " + ", ".join([f"`{a}`" for a in legal]) if legal else "⚠️ (not your turn)"
        
        winner_text = self.hand.winner_text() if self.hand.done else ""
        md = render_state_md(obs0, legal_mask, self.hand.done, winner_text, stacks_text=self.stacks_text())

        return md, legal_hint, board_g, hero_g, opp_g, "\n".join(self.history)


# -----------------------------
# Local model path
# -----------------------------
MODEL_PATH = r"D:\RL Project\models\SP-U20_w-OppM_lr0.0003_final.pth"
AGENT_POLICY = AgentPolicy(MODEL_PATH, device="cpu")


# -----------------------------
# Gradio callbacks
# -----------------------------
def _parse_seed(seed_text):
    seed = None
    if seed_text is not None:
        s = str(seed_text).strip()
        if s != "":
            try:
                seed = int(float(s))
            except ValueError:
                seed = None
    return seed


def start_match(initial_stack):
    ms = MatchState(AGENT_POLICY, initial_stack=int(float(initial_stack)))
    ms.history.append(f"=== New Match (initial_stack={int(float(initial_stack))}) ===")
    ms.start_new_hand(seed=None) 

    if ms.hand.done:
        ms.apply_payout_if_done()
    
    return ms, *ms.view_outputs()

def new_hand(match_state, seed_text):
    if match_state is None:
        match_state = MatchState(AGENT_POLICY, initial_stack=DEFAULT_INITIAL_STACK)
        match_state.history.append(f"=== New Match (initial_stack={DEFAULT_INITIAL_STACK}) ===")

    seed = _parse_seed(seed_text)
    match_state.start_new_hand(seed=seed)

    if match_state.hand.done:
        match_state.apply_payout_if_done()

    return match_state, *match_state.view_outputs()


def do_action(match_state, action_int):
    if match_state is None:
        match_state = MatchState(AGENT_POLICY, initial_stack=DEFAULT_INITIAL_STACK)
        match_state.history.append(f"=== New Match (initial_stack={DEFAULT_INITIAL_STACK}) ===")
        match_state.start_new_hand(seed=None)

    if match_state.hand is None:
        match_state.start_new_hand(seed=None)

    # Step once (this may trigger multiple agent actions)
    match_state.hand.step_human(int(action_int))

    # IMPORTANT: append ALL new action lines (including opponent actions)
    match_state.sync_new_hand_log_lines()

    # If hand ended, update bankroll and log it
    if match_state.hand.done:
        match_state.apply_payout_if_done()

    return match_state, *match_state.view_outputs()


# -----------------------------
# Gradio UI
# -----------------------------
CUSTOM_CSS = """
.gradio-container .gallery img { object-fit: contain !important; }
.gradio-container .gallery { overflow: hidden !important; }
"""
CARD_HEIGHT = 400

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# Heads-Up Texas Hold'em (PettingZoo) — Local Human vs RL Agent (Persistent Match)")

    match_state = gr.State(None)

    with gr.Row():
        stack_in = gr.Number(label="Initial chips (each) — for NEW MATCH", value=DEFAULT_INITIAL_STACK, precision=0)
        start_match_btn = gr.Button("Start New Match", variant="primary")

    with gr.Row():
        seed_in = gr.Textbox(label="Seed (optional) — for next hand only", placeholder="leave blank for random")
        new_hand_btn = gr.Button("New Hand (keep chips + history)", variant="secondary")

    table_md = gr.Markdown()

    gr.Markdown("## Community Cards")
    board_gallery = gr.Gallery(label="Board", columns=5, rows=1, height=CARD_HEIGHT, object_fit="none", show_label=True)

    gr.Markdown("## Hole Cards")
    with gr.Row():
        hero_gallery = gr.Gallery(label="You (player_0)", columns=2, rows=1, height=CARD_HEIGHT, object_fit="none", show_label=True)
        opp_gallery = gr.Gallery(label="Opponent (player_1)", columns=2, rows=1, height=CARD_HEIGHT, object_fit="none", show_label=True)

    gr.Markdown("### Your actions (player_0)")
    action_hint = gr.Markdown("Please start a match to see legal actions.")
    with gr.Row():
        btn_call = gr.Button("Call (0)")
        btn_raise = gr.Button("Raise (1)")
        btn_fold = gr.Button("Fold (2)")
        btn_check = gr.Button("Check (3)")

    history = gr.Textbox(label="Match History (full, all hands)", lines=22)

    start_match_btn.click(
        fn=start_match,
        inputs=[stack_in],
        outputs=[match_state, table_md, action_hint, board_gallery, hero_gallery, opp_gallery, history],
    )

    new_hand_btn.click(
        fn=new_hand,
        inputs=[match_state, seed_in],
        outputs=[match_state, table_md, action_hint, board_gallery, hero_gallery, opp_gallery, history], # 增加 action_hint
    )

    btn_call.click(fn=lambda ms: do_action(ms, 0), inputs=[match_state],
                   outputs=[match_state, table_md, action_hint, board_gallery, hero_gallery, opp_gallery, history])
    
    btn_raise.click(fn=lambda ms: do_action(ms, 1), inputs=[match_state],
                    outputs=[match_state, table_md, action_hint, board_gallery, hero_gallery, opp_gallery, history])
    
    btn_fold.click(fn=lambda ms: do_action(ms, 2), inputs=[match_state],
                   outputs=[match_state, table_md, action_hint, board_gallery, hero_gallery, opp_gallery, history])
    
    btn_check.click(fn=lambda ms: do_action(ms, 3), inputs=[match_state],
                    outputs=[match_state, table_md, action_hint, board_gallery, hero_gallery, opp_gallery, history])


if __name__ == "__main__":
    if not os.path.isdir(CARD_DIR):
        print(f"[WARN] Card image folder not found: {CARD_DIR}")
    if not os.path.exists(BACK_PATH):
        print(f"[WARN] Missing back-card image: {BACK_PATH}")
        print("       Add poker_cards/BACK.png as the unrevealed card image.")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False, allowed_paths=[CARD_DIR])