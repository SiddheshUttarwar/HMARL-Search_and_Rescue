"""
SAR Grid Environment (Dec-POMDP) with random episode generation.
"""
import numpy as np
from config import Config
from belief import BeliefMap


class SAREnvironment:
    """
    2-D grid search-and-rescue environment.
    • Agents move {up, down, left, right, stay}.
    • Victims are placed randomly; agents detect them within detection_radius.
    • Observations: local belief window (flattened) + normalised own position.
    """

    ACTION_MAP = {0: (0, -1),   # up
                  1: (0, 1),    # down
                  2: (-1, 0),   # left
                  3: (1, 0),    # right
                  4: (0, 0)}    # stay

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.belief = BeliefMap(cfg.grid_width, cfg.grid_height, cfg.n_victims)
        # State variables (set in reset)
        self.agent_pos: np.ndarray | None = None
        self.victim_pos: np.ndarray | None = None
        self.victims_found: np.ndarray | None = None
        self.step_count: int = 0

    # ── Reset (random episode generation) ────────────────────────
    def reset(self) -> np.ndarray:
        """Generate a fresh random episode and return initial observations."""
        cfg = self.cfg
        # Random agent spawn positions
        self.agent_pos = np.column_stack([
            self.rng.randint(0, cfg.grid_width, size=cfg.n_agents),
            self.rng.randint(0, cfg.grid_height, size=cfg.n_agents),
        ]).astype(np.float64)

        # Random victim positions (non-overlapping with agents)
        victims = []
        while len(victims) < cfg.n_victims:
            vx = self.rng.randint(0, cfg.grid_width)
            vy = self.rng.randint(0, cfg.grid_height)
            victims.append((vx, vy))
        self.victim_pos = np.array(victims, dtype=np.float64)

        self.victims_found = np.zeros(cfg.n_victims, dtype=bool)
        self.step_count = 0
        self.belief.reset(cfg.n_victims)

        # Initial belief update based on spawn positions
        self.belief.update(self.agent_pos, self.victim_pos, cfg.detection_radius)
        return self._get_observations()

    # ── Step ─────────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        """
        actions: int array of shape (n_agents,) in {0..4}.
        Returns: (observations, low_level_rewards, done, info)
        """
        cfg = self.cfg
        # Move agents
        for i, a in enumerate(actions):
            dx, dy = self.ACTION_MAP[int(a)]
            nx = np.clip(self.agent_pos[i, 0] + dx, 0, cfg.grid_width - 1)
            ny = np.clip(self.agent_pos[i, 1] + dy, 0, cfg.grid_height - 1)
            self.agent_pos[i] = [nx, ny]

        # Belief update
        old_potential = self.belief.potential()
        self.belief.update(self.agent_pos, self.victim_pos, cfg.detection_radius)
        new_potential = self.belief.potential()

        # Check victim detection
        for vi, (vx, vy) in enumerate(self.victim_pos):
            if self.victims_found[vi]:
                continue
            for ai in range(cfg.n_agents):
                dist = np.sqrt((self.agent_pos[ai, 0] - vx) ** 2 +
                               (self.agent_pos[ai, 1] - vy) ** 2)
                if dist <= cfg.detection_radius:
                    self.victims_found[vi] = True

        self.step_count += 1
        done = bool(np.all(self.victims_found) or
                     self.step_count >= cfg.max_episode_steps)

        # Low-level reward: small per-agent entropy reduction share
        reward_per_agent = np.full(cfg.n_agents,
                                   (new_potential - old_potential) / cfg.n_agents,
                                   dtype=np.float64)

        info = {
            "potential": new_potential,
            "entropy": self.belief.entropy(),
            "victims_found": int(np.sum(self.victims_found)),
            "step": self.step_count,
        }
        return self._get_observations(), reward_per_agent, done, info

    # ── Observations ─────────────────────────────────────────────
    def _get_observations(self) -> np.ndarray:
        """
        Returns obs array of shape (n_agents, obs_dim).
        obs_i = [local_belief_flat, norm_x, norm_y]
        """
        cfg = self.cfg
        obs_list = []
        for i in range(cfg.n_agents):
            x, y = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            local = self.belief.get_local_belief(x, y, cfg.obs_radius).flatten()
            pos = np.array([x / cfg.grid_width, y / cfg.grid_height])
            obs_list.append(np.concatenate([local, pos]))
        return np.array(obs_list, dtype=np.float32)

    # ── Helpers ──────────────────────────────────────────────────
    def get_agent_positions(self) -> np.ndarray:
        return self.agent_pos.copy()

    def get_distances_to_victims(self) -> np.ndarray:
        """For each agent, min distance to an unfound victim."""
        cfg = self.cfg
        dists = np.full(cfg.n_agents, float("inf"))
        for i in range(cfg.n_agents):
            for vi, (vx, vy) in enumerate(self.victim_pos):
                if not self.victims_found[vi]:
                    d = np.sqrt((self.agent_pos[i, 0] - vx) ** 2 +
                                (self.agent_pos[i, 1] - vy) ** 2)
                    dists[i] = min(dists[i], d)
        return dists
