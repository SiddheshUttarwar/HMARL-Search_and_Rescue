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
        
        # ── Generate fixed obstacles for this environment ────────────
        self.obstacles = np.zeros((cfg.grid_height, cfg.grid_width), dtype=bool)
        self._generate_obstacles()
        self.walkable_cells = (cfg.grid_width * cfg.grid_height) - cfg.n_obstacles
        self.belief.set_obstacles(self.obstacles)
        
        # State variables (set in reset)
        self.agent_pos: np.ndarray | None = None
        self.victim_pos: np.ndarray | None = None
        self.victims_found: np.ndarray | None = None
        self.step_count: int = 0

    # ── Obstacles ────────────────────────────────────────────────
    def _generate_obstacles(self):
        """Generate connected wall segments for realistic terrain."""
        cfg = self.cfg
        placed = 0
        seeds = 4 + self.rng.randint(0, 4)
        for _ in range(seeds):
            if placed >= cfg.n_obstacles: break
            sx = 2 + self.rng.randint(0, cfg.grid_width - 4)
            sy = 2 + self.rng.randint(0, cfg.grid_height - 4)
            length = 3 + self.rng.randint(0, 6)
            horizontal = self.rng.rand() < 0.5
            for k in range(length):
                if placed >= cfg.n_obstacles: break
                wx = sx + k if horizontal else sx
                wy = sy if horizontal else sy + k
                if 0 <= wx < cfg.grid_width and 0 <= wy < cfg.grid_height:
                    if not self.obstacles[wy, wx]:
                        self.obstacles[wy, wx] = True
                        placed += 1
                # Branching
                if self.rng.rand() < 0.3 and placed < cfg.n_obstacles:
                    bx = wx + (0 if horizontal else (1 if self.rng.rand() < 0.5 else -1))
                    by = wy + ((1 if self.rng.rand() < 0.5 else -1) if horizontal else 0)
                    if 0 <= bx < cfg.grid_width and 0 <= by < cfg.grid_height:
                        if not self.obstacles[by, bx]:
                            self.obstacles[by, bx] = True
                            placed += 1
                            
        # Fill remaining
        while placed < cfg.n_obstacles:
            rx = self.rng.randint(0, cfg.grid_width)
            ry = self.rng.randint(0, cfg.grid_height)
            if not self.obstacles[ry, rx]:
                self.obstacles[ry, rx] = True
                placed += 1

    # ── Reset (random episode generation) ────────────────────────
    def reset(self) -> np.ndarray:
        """Generate a fresh random episode and return initial observations."""
        cfg = self.cfg
        # Random agent spawn positions (on walkable cells)
        agents = []
        while len(agents) < cfg.n_agents:
            ax = self.rng.randint(0, cfg.grid_width)
            ay = self.rng.randint(0, cfg.grid_height)
            if not self.obstacles[ay, ax]:
                agents.append([ax, ay])
        self.agent_pos = np.array(agents, dtype=np.float64)

        # Random victim positions (on walkable cells)
        victims = []
        while len(victims) < cfg.n_victims:
            vx = self.rng.randint(0, cfg.grid_width)
            vy = self.rng.randint(0, cfg.grid_height)
            if self.obstacles[vy, vx]: continue
            if not any(v[0] == vx and v[1] == vy for v in victims):
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
        # Move agents (respecting obstacles)
        for i, a in enumerate(actions):
            dx, dy = self.ACTION_MAP[int(a)]
            nx = np.clip(self.agent_pos[i, 0] + dx, 0, cfg.grid_width - 1)
            ny = np.clip(self.agent_pos[i, 1] + dy, 0, cfg.grid_height - 1)
            
            # Can't walk into walls
            if not self.obstacles[int(ny), int(nx)]:
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
        obs_i = [local_belief_flat, local_obstacle_flat, norm_x, norm_y]
        """
        cfg = self.cfg
        obs_list = []
        r = cfg.obs_radius
        for i in range(cfg.n_agents):
            x, y = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            
            # Local belief window
            local_belief = self.belief.get_local_belief(x, y, r).flatten()
            
            # Local obstacle window
            size = 2 * r + 1
            local_obs = np.zeros((size, size), dtype=np.float32)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = x + dx, y + dy
                    # Out of bounds treated as walls
                    if 0 <= nx < cfg.grid_width and 0 <= ny < cfg.grid_height:
                        local_obs[dy + r, dx + r] = float(self.obstacles[ny, nx])
                    else:
                        local_obs[dy + r, dx + r] = 1.0  # Treat boundaries as obstacles
            
            pos = np.array([x / cfg.grid_width, y / cfg.grid_height])
            obs_list.append(np.concatenate([local_belief, local_obs.flatten(), pos]))
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
