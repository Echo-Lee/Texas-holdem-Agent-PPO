# app.py - Dark Casino Theme with Model Selection
import os
import yaml
import numpy as np
import gradio as gr
import torch
import uuid
from pettingzoo.classic import texas_holdem_v4
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
ACTIONS = {0: "Call", 1: "Raise", 2: "Fold", 3: "Check"}
ACTION_EMOJI = {0: "📞", 1: "💰", 2: "🚫", 3: "✅"}

# Card images
CARD_DIR = os.path.join(os.path.dirname(__file__), "poker_cards")
if not os.path.exists(CARD_DIR):
    CARD_DIR = os.path.join(os.path.dirname(__file__), "public", "poker_cards")

BACK_CODE = "BACK"
BACK_PATH = os.path.join(CARD_DIR, f"{BACK_CODE}.png")

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]
DEFAULT_INITIAL_STACK = 100

# RLCard Texas Hold'em blinds
SMALL_BLIND = 1
BIG_BLIND = 2


# -----------------------------
# Model selection helpers
# -----------------------------
def get_available_models():
    """Scan models/ directory for .pt files"""
    models_dir = Path("models")
    if not models_dir.exists():
        return ["No models found"]
    models = sorted([f.name for f in models_dir.glob("*.pt")])
    return models if models else ["No models found"]


def detect_config_from_model(model_name):
    """Auto-detect config file based on model filename"""
    if "_mlp" in model_name.lower():
        config_path = "running_config_mlp.yaml"
        if Path(config_path).exists():
            print(f"[INFO] Auto-selected {config_path} for MLP model")
            return config_path
    config_path = "running_config.yaml"
    print(f"[INFO] Using {config_path} for CNN model")
    return config_path


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
    """Old method - no longer used, kept for compatibility"""
    set1 = set(cards_p1)
    board = [c for c in cards_p0 if c in set1]
    board_set = set(board)
    hole0 = [c for c in cards_p0 if c not in board_set]
    hole1 = [c for c in cards_p1 if c not in board_set]
    return board, hole0, hole1


def card_to_str(card_obj):
    """Convert RLCard card object to string like 'AS', '2H', etc."""
    rank_map = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    suit_map = {'S': 'S', 'H': 'H', 'D': 'D', 'C': 'C'}

    rank = card_obj.rank
    suit = card_obj.suit

    if rank in rank_map:
        rank_str = rank_map[rank]
    elif rank == 10:
        rank_str = 'T'
    else:
        rank_str = str(rank)

    return rank_str + suit_map[suit]


def render_state_md(obs_h, legal_mask, done, winner_text, stacks_text, pot=0, to_call=0):
    raises = decode_round_raises(obs_h["observation"])
    raises_str = ", ".join([str(x) if x is not None else "-" for x in raises])

    if done:
        result_html = f"""
<div style='background: #1e3a20; padding: 8px; border-radius: 6px; margin: 6px 0;
            border: 2px solid #FFD700; text-align: center;'>
    <div style='color: #FFD700; font-weight: bold; font-size: 14px;'>🏆 {winner_text}</div>
</div>
"""
    else:
        result_html = ""

    # Pot and current bet display
    pot_html = f"""
<div style='background: rgba(255,215,0,0.15); padding: 8px; border-radius: 6px; margin: 6px 0;
            border: 1px solid #FFD700;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div style='flex: 1;'>
            <div style='color: #9CA3AF; font-size: 10px;'>Pot</div>
            <div style='color: #FFD700; font-size: 16px; font-weight: bold;'>💎 {pot:.0f}</div>
        </div>
        <div style='flex: 1; text-align: right;'>
            <div style='color: #9CA3AF; font-size: 10px;'>To Call</div>
            <div style='color: #60A5FA; font-size: 16px; font-weight: bold;'>💵 {to_call:.0f}</div>
        </div>
    </div>
</div>
"""

    return f"""
<div style='background: linear-gradient(135deg, #0f1e0f 0%, #1a2e1a 100%);
            padding: 8px; border-radius: 8px; border: 2px solid #2d5a2d;'>

    <div style='color: #90EE90; font-size: 13px; font-weight: bold; margin-bottom: 6px;'>
        💰 {stacks_text}
    </div>

    {pot_html}

    <div style='background: rgba(0,0,0,0.3); padding: 6px; border-radius: 6px; margin: 6px 0;'>
        <div style='color: #9CA3AF; font-size: 10px;'>Betting Rounds:</div>
        <div style='color: #D1D5DB; font-size: 12px; font-family: monospace;'>{raises_str}</div>
    </div>

    {result_html}
</div>
"""


# -----------------------------
# Agent wrapper
# -----------------------------
class AgentPolicy:
    def __init__(self, model_path: str, config_path: str = "", device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path

        # Auto-detect config if not provided
        if not config_path:
            config_path = detect_config_from_model(model_path)

        self.config_path = config_path

        obs_dim = 72  # Texas Hold'em observation
        obs_dim += 4  # opponent action encoding

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_type = cfg.get('system', {}).get('model_type', 'cnn')
        agent_cfg = cfg.get('model', {}).get('agent', {})

        # Create network based on model type
        if model_type == 'mlp':
            from core.networks.policy_value_network import FlexibleNet
            mlp_hidden = agent_cfg.get('mlp_hidden_layers', [256, 256, 128])
            use_ln = agent_cfg.get('use_layer_norm', True)
            self.policy_net = FlexibleNet(
                input_dim=obs_dim,
                output_dim=4,
                hidden_layers=mlp_hidden,
                use_layer_norm=use_ln
            ).to(self.device)
        else:  # CNN
            from core.networks.policy_value_network import PolicyNet
            hidden_layers = agent_cfg.get('hidden_layers', [256, 256, 128])
            card_channels = agent_cfg.get('card_encoder_channels', [16, 32])
            card_embed = agent_cfg.get('card_embedding_dim', 128)
            use_ln = agent_cfg.get('use_layer_norm', True)
            self.policy_net = PolicyNet(
                input_dim=obs_dim,
                output_dim=4,
                hidden_layers=hidden_layers,
                use_layer_norm=use_ln,
                card_encoder_channels=card_channels,
                card_embedding_dim=card_embed
            ).to(self.device)

        self.load(model_path)

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "policy_state_dict" in state:
            state = state["policy_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.policy_net.load_state_dict(state, strict=False)
            print(f"[OK] Loaded model: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")

    @torch.no_grad()
    def act(self, obs_dict):
        obs = obs_dict["observation"].astype(np.float32)
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
# Poker Hand & Match State
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
        self.action_log = []
        self.action_log.append(f"🎲 New Hand (seed={self.seed})")
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

                # Track bet amount for Raise actions
                action_text = ACTIONS[a]
                if a == 1:  # Raise action
                    try:
                        game = self.env.unwrapped.env.game
                        player_idx = 1  # Agent is player_1
                        before_bet = game.players[player_idx].in_chips
                        self._step_checked(a)
                        after_bet = game.players[player_idx].in_chips
                        raise_amount = after_bet - before_bet
                        action_text = f"{ACTIONS[a]} {raise_amount:.0f} chips"
                    except Exception:
                        self._step_checked(a)
                else:
                    self._step_checked(a)

                self.action_log.append(f"🤖 Agent: {ACTION_EMOJI[a]} {action_text}")
                continue
            if agent == self.human:
                break
            self.env.step(None)

    def step_human(self, action_int):
        if self.done:
            self.action_log.append("⚠️ Hand is over.")
            return
        if self._current_agent() != self.human:
            self._auto_play_until_human()
        if self.done:
            return
        obs = self._obs(self.human)
        mask = obs["action_mask"].astype(int)
        if mask[action_int] != 1:
            self.action_log.append(f"❌ You: ILLEGAL {ACTIONS[action_int]}")
            self._step_checked(action_int)
            self._auto_play_until_human()
        else:
            # Track bet amount for Raise actions
            action_text = ACTIONS[action_int]
            if action_int == 1:  # Raise action
                try:
                    game = self.env.unwrapped.env.game
                    player_idx = 0  # Human is player_0
                    before_bet = game.players[player_idx].in_chips
                    self._step_checked(action_int)
                    after_bet = game.players[player_idx].in_chips
                    raise_amount = after_bet - before_bet
                    action_text = f"{ACTIONS[action_int]} {raise_amount:.0f} chips"
                except Exception:
                    self._step_checked(action_int)
            else:
                self._step_checked(action_int)

            self.action_log.append(f"👤 You: {ACTION_EMOJI[action_int]} {action_text}")
            self._auto_play_until_human()

    def rewards(self):
        return float(self.env.rewards.get(self.human, 0.0)), float(self.env.rewards.get(self.bot, 0.0))

    def winner_text(self):
        r0, r1 = self.rewards()
        # Convert normalized rewards to actual chips
        chip0 = r0 * BIG_BLIND
        chip1 = r1 * BIG_BLIND

        eps = 1e-9
        if r0 > r1 + eps:
            return f"You Win! (+{chip0:.1f} chips)"
        elif r1 > r0 + eps:
            return f"Agent Wins ({chip0:+.1f} chips)"
        return f"Push (Tie)"

    def get_pot_info(self):
        """Get current pot and bet information from the environment"""
        try:
            # Access RLCard game state through PettingZoo wrapper
            game = self.env.unwrapped.env.game

            # Total pot across the entire hand (not just current round)
            pot = sum(p.in_chips for p in game.players)

            # Get current bet to call (difference between max bet and player's bet)
            player_idx = 0 if self.human == "player_0" else 1
            player_bets = [p.in_chips for p in game.players]

            max_bet = max(player_bets)
            player_bet = player_bets[player_idx]
            to_call = max_bet - player_bet

            return pot, to_call
        except Exception:
            # Fallback if environment structure is different
            return 0, 0

    def galleries(self):
        obs0 = self._obs(self.human)

        try:
            # Get cards directly from RLCard game object to preserve order
            game = self.env.unwrapped.env.game

            # Public cards (board) - in the order they were dealt
            board = [card_to_str(c) for c in game.public_cards] if hasattr(game, 'public_cards') else []

            # Player hands
            player_0_hand = [card_to_str(c) for c in game.players[0].hand] if hasattr(game.players[0], 'hand') else []
            player_1_hand = [card_to_str(c) for c in game.players[1].hand] if hasattr(game.players[1], 'hand') else []

            hole0 = player_0_hand  # Human's cards
            hole1 = player_1_hand  # Agent's cards
        except Exception:
            # Fallback to old method if direct access fails
            cards0 = decode_cards(obs0["observation"])
            obs1 = self._obs(self.bot)
            cards1 = decode_cards(obs1["observation"])
            board, hole0, hole1 = split_board_and_holes_preserve_order(cards0, cards1)

        board_display = pad_with_back(board, 5)
        hero_display = pad_with_back(hole0, 2)
        opp_display = hole1 if self.done else [BACK_CODE, BACK_CODE]
        opp_display = pad_with_back(opp_display, 2)
        pot, to_call = self.get_pot_info()
        return (
            cards_to_gallery_items(board_display),
            cards_to_gallery_items(hero_display),
            cards_to_gallery_items(opp_display),
            obs0["action_mask"].astype(int),
            obs0,
            pot,
            to_call,
        )


class MatchState:
    def __init__(self, agent_policy: AgentPolicy, initial_stack=DEFAULT_INITIAL_STACK):
        self.agent_policy = agent_policy
        self.initial_stack = float(initial_stack)
        self.stack = {"player_0": float(initial_stack), "player_1": float(initial_stack)}
        self.hand_no = 0
        self.history = []
        self.hand = None
        self.payout_applied = False
        self.hand_log_pos = 0

    def stacks_text(self):
        return f"You: {self.stack['player_0']:.1f} chips | Agent: {self.stack['player_1']:.1f} chips"

    def start_new_hand(self, seed=None):
        self.apply_payout_if_done()
        self.hand_no += 1
        self.payout_applied = False
        self.history.append("\n" + "="*50)
        self.history.append(f"🎴 Hand #{self.hand_no}")
        self.hand = PokerHand(self.agent_policy, seed=seed)
        self.history.extend(self.hand.action_log)
        self.hand_log_pos = len(self.hand.action_log)

    def sync_new_hand_log_lines(self):
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

        # Get rewards (normalized by big blind) and convert to chips
        r0, r1 = self.hand.rewards()
        chip0 = r0 * BIG_BLIND
        chip1 = r1 * BIG_BLIND

        self.stack["player_0"] += chip0
        self.stack["player_1"] += chip1
        self.history.append(f"💸 Payout: You {chip0:+.1f} | Agent {chip1:+.1f} chips")
        self.history.append(f"💰 Stacks: You {self.stack['player_0']:.1f} | Agent {self.stack['player_1']:.1f}")
        self.payout_applied = True

    def view_outputs(self):
        if self.hand is None:
            return (
                f"<div style='text-align: center; padding: 40px;'><h2 style='color: #FFD700;'>🎰 Welcome to Texas Hold'em</h2><p style='color: #9CA3AF;'>{self.stacks_text()}</p><p style='color: #60A5FA;'>Select a model and click 'New Match' to start!</p></div>",
                "\n".join(self.history) if self.history else "Ready to play...",
                [], [], [],
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                ""
            )
        board_g, hero_g, opp_g, legal_mask, obs0, pot, to_call = self.hand.galleries()
        winner_text = self.hand.winner_text() if self.hand.done else ""
        md = render_state_md(obs0, legal_mask, self.hand.done, winner_text, stacks_text=self.stacks_text(), pot=pot, to_call=to_call)

        # Update button states based on legal actions
        btn_call_update = gr.update(interactive=bool(legal_mask[0]))
        btn_raise_update = gr.update(interactive=bool(legal_mask[1]))
        btn_fold_update = gr.update(interactive=bool(legal_mask[2]))
        btn_check_update = gr.update(interactive=bool(legal_mask[3]))

        # Chips display
        chips_html = f"<div style='color: #FFD700; font-size: 13px; font-weight: bold; text-align: center; padding: 3px; margin: 3px 0;'>💰 Your Chips: {self.stack['player_0']:.1f}</div>"

        return md, "\n".join(self.history), board_g, hero_g, opp_g, btn_call_update, btn_raise_update, btn_fold_update, btn_check_update, chips_html


# -----------------------------
# Initialize agent (lazy loading)
# -----------------------------
_sessions = {}  # Server-side session storage: {session_id: MatchState}


def get_session(session_id):
    """Get or create a session's MatchState"""
    if session_id not in _sessions:
        return None
    return _sessions[session_id]


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


def start_match(session_id, model_name, initial_stack):
    """Start a new match with selected model"""
    if model_name == "No models found" or not model_name:
        return (
            session_id,
            "<div style='text-align: center; padding: 20px; color: #EF4444;'>⚠️ Please select a valid model!</div>",
            "No model selected",
            [], [], [],
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
            ""
        )

    try:
        model_path = f"models/{model_name}"
        agent = AgentPolicy(model_path, device="cpu")
        ms = MatchState(agent, initial_stack=int(float(initial_stack)))
        ms.history.append(f"🎰 New Match Started (starting chips: {int(float(initial_stack))})")
        ms.history.append(f"🤖 Opponent: {model_name}")
        ms.start_new_hand(seed=None)
        if ms.hand and ms.hand.done:
            ms.apply_payout_if_done()
        _sessions[session_id] = ms  # Store in server-side storage
        return session_id, *ms.view_outputs()
    except Exception as e:
        return (
            session_id,
            f"<div style='text-align: center; padding: 20px; color: #EF4444;'>❌ Error loading model: {str(e)}</div>",
            f"Error: {str(e)}",
            [], [], [],
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
            ""
        )


def new_hand(session_id, seed_text):
    ms = get_session(session_id)
    if ms is None:
        return (
            session_id,
            "<div style='text-align: center; padding: 20px; color: #EF4444;'>⚠️ Please start a match first!</div>",
            "No active match",
            [], [], [],
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
            ""
        )
    seed = _parse_seed(seed_text)
    ms.start_new_hand(seed=seed)
    if ms.hand.done:
        ms.apply_payout_if_done()
    return session_id, *ms.view_outputs()


def do_action(session_id, action_int):
    ms = get_session(session_id)
    if ms is None or ms.hand is None:
        return (
            session_id,
            "<div style='text-align: center; padding: 20px; color: #EF4444;'>⚠️ Please start a match first!</div>",
            "No active match",
            [], [], [],
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
            ""
        )
    ms.hand.step_human(int(action_int))
    ms.sync_new_hand_log_lines()
    if ms.hand.done:
        ms.apply_payout_if_done()
    return session_id, *ms.view_outputs()


# -----------------------------
# Dark Casino Theme & UI
# -----------------------------
CUSTOM_CSS = """
/* Dark Casino Theme - Compact Layout */
.gradio-container {
    background: linear-gradient(135deg, #0a0f0a 0%, #1a1f1a 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-height: 100vh !important;
    overflow-y: auto !important;
}

/* Remove excessive padding */
.gradio-container .block {
    padding: 2px !important;
    margin: 2px 0 !important;
}

/* Compact rows */
.gradio-container .row {
    gap: 6px !important;
    margin: 2px 0 !important;
}

/* Compact columns */
.gradio-container .column {
    padding: 2px !important;
}

/* Card gallery styling - compact with forced size limits */
.gradio-container .gallery {
    height: 85px !important;
    min-height: 85px !important;
    max-height: 85px !important;
    background: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid #2d5a2d !important;
    border-radius: 6px !important;
    margin: 1px 0 !important;
    display: flex !important;
    justify-content: center !important;
}

/* Force absolute size limits on high-res card images */
.gradio-container .gallery img {
    height: 70px !important;
    width: auto !important;
    max-width: 50px !important;
    object-fit: contain !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5) !important;
    transition: transform 0.2s ease !important;
}

.gradio-container .gallery img:hover {
    transform: scale(1.03) !important;
}

/* Remove internal padding that pushes large images out */
.gradio-container .gallery .grid-wrap {
    padding: 2px !important;
}

/* Button styling - Casino gold theme - More compact */
.gradio-container button {
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
    color: #000 !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0px 10px !important;
    font-size: 13px !important;
    box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3) !important;
    transition: all 0.2s ease !important;
    min-height: 30px !important;
    height: 30px !important;
}

.gradio-container button:hover:not(:disabled) {
    background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(255, 215, 0, 0.5) !important;
}

.gradio-container button:active:not(:disabled) {
    transform: translateY(0px) !important;
}

/* Disabled buttons - grayed out */
.gradio-container button:disabled {
    background: linear-gradient(135deg, #4a4a4a 0%, #2a2a2a 100%) !important;
    color: #666 !important;
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

/* Primary button (Start Match / Deal Hand) */
.gradio-container button.primary:not(:disabled) {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: white !important;
}

.gradio-container button.primary:hover:not(:disabled) {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
}

/* Secondary button (New Hand) */
.gradio-container button.secondary:not(:disabled) {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: white !important;
}

/* Input fields - compact */
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 1px solid #2d5a2d !important;
    color: #E5E7EB !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    padding: 4px 6px !important;
    min-height: 32px !important;
}

/* Dropdown styling */
.gradio-container select {
    cursor: pointer !important;
}

/* Labels - smaller */
.gradio-container label {
    color: #9CA3AF !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    margin-bottom: 2px !important;
}

/* Number inputs */
.gradio-container input[type="number"] {
    min-height: 32px !important;
    height: 32px !important;
}

/* Compact input styling */
.compact-input input, .compact-input select {
    height: 32px !important;
    min-height: 32px !important;
    padding: 4px 6px !important;
}

/* Markdown content */
.gradio-container .markdown {
    color: #E5E7EB !important;
}

/* Textbox for history - compact */
.gradio-container textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    line-height: 1.2 !important;
    padding: 4px !important;
}

/* Header styling - compact */
h1, h2, h3 {
    color: #FFD700 !important;
    text-shadow: 0 0 10px rgba(255, 215, 0, 0.3) !important;
    margin: 2px 0 !important;
    line-height: 1.2 !important;
}
"""

CARD_HEIGHT = 85  # Compact card size matching CSS constraints

# Create custom theme
casino_theme = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    font=("Helvetica", "Arial", "sans-serif"),
).set(
    body_background_fill="linear-gradient(135deg, #0a0f0a 0%, #1a1f1a 100%)",
    block_background_fill="*neutral_950",
    block_border_color="*neutral_800",
    input_background_fill="*neutral_900",
)

with gr.Blocks(css=CUSTOM_CSS, theme=casino_theme, title="🎰 Texas Hold'em AI") as demo:
    # Session-based storage (serializable for Gradio 5.x)
    session_id = gr.State(lambda: str(uuid.uuid4()))

    # Compact header
    gr.HTML("""
        <div style='text-align: center; padding: 6px; background: linear-gradient(135deg, #1e3a20 0%, #0f1e0f 100%);
                    border-radius: 8px; margin-bottom: 6px; border: 2px solid #FFD700;'>
            <h1 style='margin: 0; font-size: 20px; color: #FFD700;'>🎰 Texas Hold'em PPO Agent</h1>
            <div style='color: #9CA3AF; font-size: 11px; margin-top: 3px;'>Select your opponent from trained models</div>
        </div>
    """)

    # Controls in compact row
    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="🤖 Select Opponent",
            choices=get_available_models(),
            value=get_available_models()[0] if get_available_models() else None,
            scale=2,
            elem_classes="compact-input"
        )
        stack_in = gr.Number(label="💰 Chips", value=DEFAULT_INITIAL_STACK, precision=0, scale=1, elem_classes="compact-input")
        start_match_btn = gr.Button("🎲 New Match", variant="primary", scale=1, size="sm")

    with gr.Row():
        seed_in = gr.Textbox(label="🎲 Seed", placeholder="Random", scale=1, elem_classes="compact-input")
        new_hand_btn = gr.Button("🎴 Deal Hand", variant="secondary", scale=1, size="sm")

    # Main playing area - compact single view
    with gr.Row():
        with gr.Column(scale=3):
            # Agent's cards at top
            gr.HTML("<div style='text-align: center; color: #EF4444; font-weight: bold; font-size: 12px; margin: 2px 0;'>🤖 Agent's Cards</div>")
            opp_gallery = gr.Gallery(
                columns=2,
                rows=1,
                height=85,
                object_fit="contain",
                show_label=False,
                preview=False
            )

            # Community cards in center
            gr.HTML("<div style='text-align: center; color: #FFD700; font-weight: bold; font-size: 12px; margin: 5px 0 2px 0;'>🃏 Board</div>")
            board_gallery = gr.Gallery(
                columns=5,
                rows=1,
                height=85,
                object_fit="contain",
                show_label=False,
                preview=False
            )

            # Player's cards at bottom
            gr.HTML("<div style='text-align: center; color: #10B981; font-weight: bold; font-size: 12px; margin: 5px 0 2px 0;'>👤 Your Cards</div>")
            hero_gallery = gr.Gallery(
                columns=2,
                rows=1,
                height=85,
                object_fit="contain",
                show_label=False,
                preview=False
            )

            # Chips display
            chips_display = gr.HTML("<div style='color: #FFD700; font-size: 13px; font-weight: bold; text-align: center; padding: 3px; margin: 3px 0;'>💰 Your Chips: --</div>")

            # Action buttons - MUST be visible!
            with gr.Row():
                btn_call = gr.Button("📞 Call", size="sm", interactive=False)
                btn_raise = gr.Button("💰 Raise", size="sm", interactive=False)
                btn_fold = gr.Button("🚫 Fold", size="sm", interactive=False)
                btn_check = gr.Button("✅ Check", size="sm", interactive=False)

        with gr.Column(scale=1):
            # Game state info
            table_md = gr.HTML()

            # Compact action log
            gr.HTML("<div style='color: #9CA3AF; font-weight: bold; font-size: 12px; margin: 3px 0 2px 0;'>📜 Log</div>")
            history = gr.Textbox(label="", lines=13, show_label=False, max_lines=13)

    # Event handlers
    outputs_list = [session_id, table_md, history, board_gallery, hero_gallery, opp_gallery, btn_call, btn_raise, btn_fold, btn_check, chips_display]

    start_match_btn.click(
        fn=start_match,
        inputs=[session_id, model_dropdown, stack_in],
        outputs=outputs_list,
    )

    new_hand_btn.click(
        fn=new_hand,
        inputs=[session_id, seed_in],
        outputs=outputs_list,
    )

    btn_call.click(
        fn=lambda sid: do_action(sid, 0),
        inputs=[session_id],
        outputs=outputs_list
    )
    btn_raise.click(
        fn=lambda sid: do_action(sid, 1),
        inputs=[session_id],
        outputs=outputs_list
    )
    btn_fold.click(
        fn=lambda sid: do_action(sid, 2),
        inputs=[session_id],
        outputs=outputs_list
    )
    btn_check.click(
        fn=lambda sid: do_action(sid, 3),
        inputs=[session_id],
        outputs=outputs_list
    )


if __name__ == "__main__":
    if not os.path.isdir(CARD_DIR):
        print(f"[WARN] Card image folder not found: {CARD_DIR}")
    print(f"[INFO] Available models: {get_available_models()}")
    demo.launch(server_name="0.0.0.0", server_port=7860)
