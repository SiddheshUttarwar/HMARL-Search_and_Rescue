"""
Belief map: Bayesian probability grid over victim locations.
Provides entropy computation and potential function Φ(s).
"""
import numpy as np


class BeliefMap:
    """
    Maintains b_t(x) = P(victim at x | H_t) for every cell x.
    Supports Bayesian updates, entropy H(b_t), and mission potential Φ.
    """

    def __init__(self, width: int, height: int, n_victims: int):
        self.width = width
        self.height = height
        self.n_victims = n_victims
        # Prior: uniform probability (will be overridden by reset)
        self.grid = np.zeros((height, width), dtype=np.float64)
        self._confirmed = np.zeros((height, width), dtype=bool)
        self.obstacles = np.zeros((height, width), dtype=bool)
        self.walkable_cells = width * height

    def set_obstacles(self, obstacles: np.ndarray):
        """Set fixed obstacles and compute walkable cell count."""
        self.obstacles = obstacles.copy()
        self.walkable_cells = int(np.sum(~self.obstacles))

    def reset(self, n_victims: int | None = None):
        if n_victims is not None:
            self.n_victims = n_victims
        # Uniform probability only over walkable cells
        prior = self.n_victims / max(1, self.walkable_cells)
        self.grid = np.where(self.obstacles, 0.0, prior).astype(np.float64)
        self._confirmed = np.zeros((self.height, self.width), dtype=bool)

    # ── Bayesian update ──────────────────────────────────────────
    def update(self, agent_positions: np.ndarray, victim_positions: np.ndarray,
               detection_radius: int):
        """
        For each agent, cells within detection_radius are observed.
        If a victim is present → set b(x) = 1 (confirmed).
        If no victim           → set b(x) = 0 (cleared).
        """
        for ax, ay in agent_positions:
            for dx in range(-detection_radius, detection_radius + 1):
                for dy in range(-detection_radius, detection_radius + 1):
                    nx, ny = int(ax + dx), int(ay + dy)
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self._confirmed[ny, nx]:
                            continue
                        # Check if any victim is at this cell
                        found = False
                        for vx, vy in victim_positions:
                            if int(vx) == nx and int(vy) == ny:
                                found = True
                                break
                        if found:
                            self.grid[ny, nx] = 1.0
                            self._confirmed[ny, nx] = True
                        else:
                            self.grid[ny, nx] = 0.0

    # ── Entropy H(b_t) = -Σ b(x) log b(x) ───────────────────────
    def entropy(self) -> float:
        g = self.grid.copy()
        # Avoid log(0)
        g = np.clip(g, 1e-12, 1.0 - 1e-12)
        return float(-np.sum(g * np.log(g) + (1 - g) * np.log(1 - g)))

    # ── Mission potential Φ(s) = -H(b_t) ────────────────────────
    def potential(self) -> float:
        return -self.entropy()

    # ── Getters ──────────────────────────────────────────────────
    def get_local_belief(self, x: int, y: int, radius: int) -> np.ndarray:
        """Return the belief values in a (2r+1)×(2r+1) window, zero-padded."""
        size = 2 * radius + 1
        patch = np.zeros((size, size), dtype=np.float64)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    patch[dy + radius, dx + radius] = self.grid[ny, nx]
        return patch
