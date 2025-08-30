import os
import random
from typing import Dict, List, Tuple, Optional, Iterable

class ExplicitMDPEnv:
    """
    Adapter for Storm explicit MDP files (.tra, .tra.rew, .lab).
    Exposes: reset(), step(action), get_all_states(), get_actions(state).
    
    Assumptions:
      - .tra format: first non-empty line may be 'mdp'. Then lines of:
            src action dest prob
      - .tra.rew format (optional): lines of
            src action dest reward
        Missing rewards default to 0.0.
      - .lab format (optional):
            #DECLARATION
            <labels...>
            #END
            <state> <label> [<label> ...]
        Initial state is the one labeled 'init'. If absent, uses min state id.
      - State representation returned by reset()/step() is a 1-tuple: (state_id,)
        to match code that does: state = tuple(env.reset()).
    """

    def __init__(
        self,
        tra_path: str,
        rew_path: Optional[str] = None,
        lab_path: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        self.tra_path = tra_path
        self.rew_path = rew_path
        self.lab_path = lab_path
        self._rng = random.Random(seed)

        # Parsed structures
        # transitions[(s, a)] = list[(dest, prob)]
        self.transitions: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        # rewards[(s, a, d)] = r
        self.rewards: Dict[Tuple[int, int, int], float] = {}
        # states set and actions per state
        self.states: List[int] = []
        self.actions_by_state: Dict[int, List[int]] = {}
        self.initial_state: int = 0

        self._parse_files()
        self.current_state: int = self.initial_state

    # ----------------------- Parsing -----------------------

    def _parse_files(self) -> None:
        self._parse_tra(self.tra_path)
        if self.rew_path and os.path.exists(self.rew_path):
            self._parse_rew(self.rew_path)
        if self.lab_path and os.path.exists(self.lab_path):
            self._parse_lab(self.lab_path)
        else:
            # Fall back if no lab file: initial is min state id
            if self.states:
                self.initial_state = min(self.states)

        # Build actions_by_state
        actions_by_state: Dict[int, set] = {}
        for (s, a) in self.transitions.keys():
            actions_by_state.setdefault(s, set()).add(a)
        self.actions_by_state = {s: sorted(list(acts)) for s, acts in actions_by_state.items()}

        # Ensure probabilities per (s,a) are (approximately) normalized. If not, normalize defensively.
        for key, alist in self.transitions.items():
            total = sum(p for _, p in alist)
            if total <= 0:
                # degenerate; make uniform over destinations
                k = len(alist)
                if k:
                    self.transitions[key] = [(d, 1.0 / k) for d, _ in alist]
            elif abs(total - 1.0) > 1e-10:
                self.transitions[key] = [(d, p / total) for d, p in alist]

    def _parse_tra(self, path: str) -> None:
        states = set()
        transitions: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.lower() == 'mdp':
                    continue
                parts = line.split()
                if len(parts) != 4:
                    # Ignore malformed lines silently
                    continue
                s, a, d, p = parts
                s = int(s); a = int(a); d = int(d); p = float(p)
                states.add(s); states.add(d)
                transitions.setdefault((s, a), []).append((d, p))
        self.states = sorted(states)
        self.transitions = transitions

    def _parse_rew(self, path: str) -> None:
        rewards: Dict[Tuple[int, int, int], float] = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 4:
                    continue
                s, a, d, r = parts
                rewards[(int(s), int(a), int(d))] = float(r)
        self.rewards = rewards

    def _parse_lab(self, path: str) -> None:
        labels_declared = False
        initial_candidates = []
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('#DECLARATION'):
                    labels_declared = True
                    continue
                if labels_declared:
                    if line.startswith('#END'):
                        labels_declared = False
                        continue
                    # ignore label names here
                    continue
                # After #END: lines like "0 init one"
                parts = line.split()
                if not parts:
                    continue
                s = int(parts[0])
                labs = set(parts[1:])
                if 'init' in labs:
                    initial_candidates.append(s)
        if initial_candidates:
            # if multiple 'init' labels exist, pick the smallest id
            self.initial_state = min(initial_candidates)
        elif self.states:
            self.initial_state = min(self.states)

    # ----------------------- Env API -----------------------

    def _norm_state(self, state) -> int:
        # accept int, str, (int,), ["int"], etc.
        if isinstance(state, (list, tuple)) and len(state) == 1:
            state = state[0]
        if isinstance(state, str):
            try:
                state = int(state)
            except ValueError:
                raise ValueError(f"State must be convertible to int, got {state!r}")
        if not isinstance(state, int):
            raise ValueError(f"Unsupported state type: {type(state)} (value={state!r})")
        return state

    def reset(self):
        self.current_state = self.initial_state
        return (self.current_state,)

    def get_all_states(self) -> List[int]:
        return list(self.states)

    def get_actions(self, state=None) -> List[int]:
        if state is None:
            s = self.current_state
        else:
            s = self._norm_state(state)
        return self.actions_by_state.get(s, [])

    def step(self, action: int):
        s = self.current_state
        key = (s, int(action))
        if key not in self.transitions:
            raise ValueError(f"No transitions for state {s} and action {action}.")
        dests = self.transitions[key]
        # sample next state
        r = self._rng.random()
        cum = 0.0
        next_state = dests[-1][0]  # default fallback
        for d, p in dests:
            cum += p
            if r <= cum:
                next_state = d
                break
        # reward lookup
        reward = self.rewards.get((s, int(action), next_state), 0.0)
        self.current_state = next_state
        # 'done' is always False for average-reward settings
        return (self.current_state,), reward, False, {}

    # ----------------------- Introspection -----------------------

    def to_dict(self) -> dict:
        return {
            "initial_state": self.initial_state,
            "states": list(self.states),
            "actions_by_state": {int(k): list(v) for k, v in self.actions_by_state.items()},
            "num_transitions": sum(len(v) for v in self.transitions.values())
        }
