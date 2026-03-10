
    // ═══════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════
    const CFG = {
      gridW: 20, gridH: 20,
      nAgents: 6,
      nVictims: 4,
      nObstacles: 40,       // number of obstacle cells
      detectionRadius: 2,
      proximityRadius: 5,
      kMin: 8, kMax: 30,
      maxSteps: 400,
    };

    const OPTION = { EXPLORE: 0, NAVIGATE: 1, FORM: 2 };
    const OPTION_NAME = ['Explore', 'Navigate', 'Form'];
    const OPTION_CLASS = ['opt-explore', 'opt-navigate', 'opt-form'];
    const AGENT_COLORS = [
      '#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#14b8a6', '#eab308', '#6366f1', '#ef4444'
    ];
    const OPTION_GLOW = ['rgba(6,182,212,.55)', 'rgba(245,158,11,.55)', 'rgba(16,185,129,.55)'];

    // ═══════════════════════════════════════════════════════════════════
    // State
    // ═══════════════════════════════════════════════════════════════════
    let canvas, ctx;
    let cellSize;
    let agents = [], victims = [], belief = [];
    let edges = [];
    let coverage = [];      // coverage[y][x] = number of times visited
    let obstacles = [];     // obstacles[y][x] = true if blocked
    let totalCells, walkableCells;
    let step = 0, totalSwitches = 0;
    let playing = false, animId = null, lastFrame = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Initialisation
    // ═══════════════════════════════════════════════════════════════════
    function init() {
      canvas = document.getElementById('sim');
      ctx = canvas.getContext('2d');
      cellSize = canvas.width / CFG.gridW;
      resetSim();
      requestAnimationFrame(draw);
    }

    function resetSim() {
      step = 0; totalSwitches = 0;
      document.getElementById('eventLog').innerHTML = '';
      totalCells = CFG.gridW * CFG.gridH;

      // Obstacle map — generate random walls & corridors
      obstacles = [];
      for (let y = 0; y < CFG.gridH; y++) {
        obstacles[y] = new Array(CFG.gridW).fill(false);
      }
      generateObstacles();
      walkableCells = totalCells - CFG.nObstacles;

      // Belief map — uniform prior (only over walkable cells)
      belief = [];
      for (let y = 0; y < CFG.gridH; y++) {
        belief[y] = [];
        for (let x = 0; x < CFG.gridW; x++) {
          belief[y][x] = obstacles[y][x] ? 0 : CFG.nVictims / walkableCells;
        }
      }

      // Coverage map — visit counts
      coverage = [];
      for (let y = 0; y < CFG.gridH; y++) {
        coverage[y] = new Array(CFG.gridW).fill(0);
      }

      // Random victim positions (not on obstacles)
      victims = [];
      while (victims.length < CFG.nVictims) {
        const vx = Math.floor(Math.random() * CFG.gridW);
        const vy = Math.floor(Math.random() * CFG.gridH);
        if (obstacles[vy][vx]) continue;
        if (!victims.some(v => v.x === vx && v.y === vy)) {
          victims.push({ x: vx, y: vy, found: false, foundStep: -1 });
        }
      }

      // Agents — spawn on walkable cells, spread out
      agents = [];
      for (let i = 0; i < CFG.nAgents; i++) {
        let ax, ay;
        do {
          ax = Math.floor(Math.random() * CFG.gridW);
          ay = Math.floor(Math.random() * CFG.gridH);
        } while (obstacles[ay][ax]);
        agents.push({
          id: i, x: ax, y: ay,
          option: OPTION.EXPLORE,
          optionAge: 0,
          target: null,
          color: AGENT_COLORS[i % AGENT_COLORS.length],
          trail: [],
        });
      }

      // Mark initial agent positions as covered
      for (const a of agents) markCovered(a.x, a.y);

      updateBeliefFromAgents();
      buildGraph();
      logEvent('Mission started -- agents exploring around obstacles', 'log-explore');
      updateUI();
    }

    // ── Obstacle generation ─────────────────────────────────────
    function generateObstacles() {
      // Generate connected wall segments for realistic terrain
      let placed = 0;
      const seeds = 4 + Math.floor(Math.random() * 4); // 4-7 wall seeds
      for (let s = 0; s < seeds && placed < CFG.nObstacles; s++) {
        // Pick a random seed point
        let sx = 2 + Math.floor(Math.random() * (CFG.gridW - 4));
        let sy = 2 + Math.floor(Math.random() * (CFG.gridH - 4));
        // Grow a wall segment from the seed
        const len = 3 + Math.floor(Math.random() * 6);
        const horizontal = Math.random() < 0.5;
        for (let k = 0; k < len && placed < CFG.nObstacles; k++) {
          const wx = horizontal ? sx + k : sx;
          const wy = horizontal ? sy : sy + k;
          if (wx >= 0 && wy >= 0 && wx < CFG.gridW && wy < CFG.gridH && !obstacles[wy][wx]) {
            obstacles[wy][wx] = true;
            placed++;
          }
          // Occasionally branch
          if (Math.random() < 0.3 && placed < CFG.nObstacles) {
            const bx = wx + (horizontal ? 0 : (Math.random() < 0.5 ? 1 : -1));
            const by = wy + (horizontal ? (Math.random() < 0.5 ? 1 : -1) : 0);
            if (bx >= 0 && by >= 0 && bx < CFG.gridW && by < CFG.gridH && !obstacles[by][bx]) {
              obstacles[by][bx] = true;
              placed++;
            }
          }
        }
      }
      // Fill remaining with scattered single blocks
      while (placed < CFG.nObstacles) {
        const rx = Math.floor(Math.random() * CFG.gridW);
        const ry = Math.floor(Math.random() * CFG.gridH);
        if (!obstacles[ry][rx]) {
          obstacles[ry][rx] = true;
          placed++;
        }
      }
    }

    function isBlocked(x, y) {
      if (x < 0 || y < 0 || x >= CFG.gridW || y >= CFG.gridH) return true;
      return obstacles[y][x];
    }

    function markCovered(ax, ay) {
      // Mark cells within detection radius as covered
      for (let dy = -CFG.detectionRadius; dy <= CFG.detectionRadius; dy++) {
        for (let dx = -CFG.detectionRadius; dx <= CFG.detectionRadius; dx++) {
          const nx = ax + dx, ny = ay + dy;
          if (nx >= 0 && ny >= 0 && nx < CFG.gridW && ny < CFG.gridH) {
            coverage[ny][nx]++;
          }
        }
      }
    }

    function getCoverageStats() {
      let covered = 0, totalVisits = 0, overlapVisits = 0;
      for (let y = 0; y < CFG.gridH; y++) {
        for (let x = 0; x < CFG.gridW; x++) {
          if (obstacles[y][x]) continue; // skip obstacles
          if (coverage[y][x] > 0) {
            covered++;
            totalVisits += coverage[y][x];
            if (coverage[y][x] > 1) overlapVisits += coverage[y][x] - 1;
          }
        }
      }
      const pct = covered / walkableCells;
      const overlap = totalVisits > 0 ? overlapVisits / totalVisits : 0;
      return { covered, pct, overlap };
    }

    // ═══════════════════════════════════════════════════════════════════
    // Belief map
    // ═══════════════════════════════════════════════════════════════════
    function updateBeliefFromAgents() {
      for (const a of agents) {
        for (let dy = -CFG.detectionRadius; dy <= CFG.detectionRadius; dy++) {
          for (let dx = -CFG.detectionRadius; dx <= CFG.detectionRadius; dx++) {
            const nx = a.x + dx, ny = a.y + dy;
            if (nx < 0 || ny < 0 || nx >= CFG.gridW || ny >= CFG.gridH) continue;
            const hasVictim = victims.some(v => v.x === nx && v.y === ny && !v.found);
            if (hasVictim) {
              belief[ny][nx] = 1.0;
            } else {
              belief[ny][nx] = 0.0;
            }
          }
        }
      }
    }

    function computeEntropy() {
      let H = 0;
      for (let y = 0; y < CFG.gridH; y++)
        for (let x = 0; x < CFG.gridW; x++) {
          const p = Math.max(1e-12, Math.min(1 - 1e-12, belief[y][x]));
          H -= p * Math.log(p) + (1 - p) * Math.log(1 - p);
        }
      return H;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Graph construction
    // ═══════════════════════════════════════════════════════════════════
    function buildGraph() {
      edges = [];
      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const dx = agents[j].x - agents[i].x;
          const dy = agents[j].y - agents[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= CFG.proximityRadius) {
            edges.push([i, j, dist]);
          }
        }
      }
    }

    function getNeighbors(agentIdx) {
      const nbrs = [];
      for (const [i, j] of edges) {
        if (i === agentIdx) nbrs.push(j);
        if (j === agentIdx) nbrs.push(i);
      }
      return nbrs;
    }

    // ═══════════════════════════════════════════════════════════════════
    // High-level policy logic (simulated)
    // ═══════════════════════════════════════════════════════════════════
    function updateHighLevelPolicies() {
      for (const a of agents) {
        a.optionAge++;
        const prevOption = a.option;

        // Check if any victim is found and nearby
        let nearestFoundVictim = null;
        let nearestFoundDist = Infinity;
        for (const v of victims) {
          if (!v.found) continue;
          const d = Math.sqrt((a.x - v.x) ** 2 + (a.y - v.y) ** 2);
          if (d < nearestFoundDist) { nearestFoundDist = d; nearestFoundVictim = v; }
        }

        // Check if any unfound victim is detected (in belief)
        let nearestDetected = null;
        let nearestDetDist = Infinity;
        for (const v of victims) {
          if (v.found) continue;
          if (belief[v.y][v.x] >= 0.99) {
            const d = Math.sqrt((a.x - v.x) ** 2 + (a.y - v.y) ** 2);
            if (d < nearestDetDist) { nearestDetDist = d; nearestDetected = v; }
          }
        }

        // Termination check (graph-conditioned)
        const shouldTerminate = a.optionAge >= CFG.kMax ||
          (a.optionAge >= CFG.kMin && decideTermination(a));

        if (shouldTerminate) {
          // Decide new option based on state
          if (nearestDetected && nearestDetDist > 2) {
            // Navigate to detected but unfound victim
            a.option = OPTION.NAVIGATE;
            a.target = { x: nearestDetected.x, y: nearestDetected.y };
          } else if (nearestDetected && nearestDetDist <= 2) {
            // Close to victim → Form
            a.option = OPTION.FORM;
            a.target = { x: nearestDetected.x, y: nearestDetected.y };
          } else if (nearestFoundVictim && nearestFoundDist <= 4) {
            // Near a found victim — spread out to explore more
            a.option = OPTION.EXPLORE;
            a.target = null;
          } else {
            a.option = OPTION.EXPLORE;
            a.target = null;
          }

          // Propagate through graph: neighbors adopt Navigate if victim detected
          if (a.option === OPTION.NAVIGATE || a.option === OPTION.FORM) {
            const nbrs = getNeighbors(a.id);
            for (const ni of nbrs) {
              const nb = agents[ni];
              if (nb.option === OPTION.EXPLORE && nearestDetected) {
                nb.option = OPTION.NAVIGATE;
                nb.target = { x: nearestDetected.x, y: nearestDetected.y };
                nb.optionAge = 0;
                totalSwitches++;
                logEvent(`A${nb.id} switched to Navigate (graph signal from A${a.id})`, 'log-navigate');
              }
            }
          }

          if (a.option !== prevOption) {
            totalSwitches++;
            a.optionAge = 0;
            logEvent(`A${a.id}: ${OPTION_NAME[prevOption]} -> ${OPTION_NAME[a.option]}`,
              a.option === 0 ? 'log-explore' : a.option === 1 ? 'log-navigate' : 'log-form');
          }
        }
      }
    }

    function decideTermination(agent) {
      // Simulated graph-conditioned termination
      const nbrs = getNeighbors(agent.id);
      // Higher termination probability if neighbors have different options
      let diffCount = 0;
      for (const ni of nbrs) {
        if (agents[ni].option !== agent.option) diffCount++;
      }
      const baseBeta = 0.08;
      const graphBonus = nbrs.length > 0 ? 0.15 * (diffCount / nbrs.length) : 0;
      return Math.random() < (baseBeta + graphBonus);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Low-level movement
    // ═══════════════════════════════════════════════════════════════════
    function moveAgents() {
      for (const a of agents) {
        let dx = 0, dy = 0;

        if (a.option === OPTION.EXPLORE) {
          // ── Coverage-maximizing exploration ──
          // Score each direction: prefer uncovered cells, avoid overlap,
          // repel from other exploring agents to spread out.
          dx = 0; dy = 0;
          let bestScore = -Infinity;
          const dirs = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]];
          for (const [ddx, ddy] of dirs) {
            const nx = a.x + ddx, ny = a.y + ddy;
            if (isBlocked(nx, ny)) continue;  // obstacle-aware: skip blocked cells

            // Prefer uncovered cells (high belief = unexplored)
            let score = belief[ny][nx] * 3.0;

            // Penalize already-covered cells (coverage overlap penalty)
            score -= coverage[ny][nx] * 0.8;

            // Bonus for being adjacent to obstacles (explore edges of walls)
            for (const [odx, ody] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
              if (isBlocked(nx + odx, ny + ody)) { score += 0.15; break; }
            }

            // Repel from other exploring agents (spread out for max coverage)
            for (const other of agents) {
              if (other.id === a.id) continue;
              if (other.option !== OPTION.EXPLORE) continue;
              const od = Math.sqrt((nx - other.x) ** 2 + (ny - other.y) ** 2);
              if (od < 4) score += od * 0.4;  // reward distance from others
            }

            // Small randomness for tie-breaking
            score += Math.random() * 0.2;

            if (score > bestScore) { bestScore = score; dx = ddx; dy = ddy; }
          }
        } else if (a.option === OPTION.NAVIGATE && a.target) {
          // Navigate toward target victim — obstacle-aware
          const tdx = a.target.x - a.x;
          const tdy = a.target.y - a.y;
          // Try primary direction first; if blocked, try alternate
          let moved = false;
          if (Math.abs(tdx) > Math.abs(tdy)) {
            if (!isBlocked(a.x + Math.sign(tdx), a.y)) { dx = Math.sign(tdx); moved = true; }
            else if (tdy !== 0 && !isBlocked(a.x, a.y + Math.sign(tdy))) { dy = Math.sign(tdy); moved = true; }
          } else {
            if (tdy !== 0 && !isBlocked(a.x, a.y + Math.sign(tdy))) { dy = Math.sign(tdy); moved = true; }
            else if (tdx !== 0 && !isBlocked(a.x + Math.sign(tdx), a.y)) { dx = Math.sign(tdx); moved = true; }
          }
          // Wall-following fallback: try perpendicular directions
          if (!moved) {
            const alts = [[0, 1], [0, -1], [1, 0], [-1, 0]];
            for (const [adx, ady] of alts) {
              if (!isBlocked(a.x + adx, a.y + ady)) { dx = adx; dy = ady; break; }
            }
          }
        } else if (a.option === OPTION.FORM && a.target) {
          // Form: move toward target — obstacle-aware
          const dist = Math.sqrt((a.x - a.target.x) ** 2 + (a.y - a.target.y) ** 2);
          if (dist > 1.5) {
            const tdx = a.target.x - a.x;
            const tdy = a.target.y - a.y;
            if (Math.abs(tdx) >= Math.abs(tdy)) {
              if (!isBlocked(a.x + Math.sign(tdx), a.y)) dx = Math.sign(tdx);
              else if (!isBlocked(a.x, a.y + Math.sign(tdy || 1))) dy = Math.sign(tdy || 1);
            } else {
              if (!isBlocked(a.x, a.y + Math.sign(tdy))) dy = Math.sign(tdy);
              else if (!isBlocked(a.x + Math.sign(tdx || 1), a.y)) dx = Math.sign(tdx || 1);
            }
          } else {
            // Hold position / circle — avoid obstacles
            const angle = (a.id / CFG.nAgents) * Math.PI * 2 + step * 0.1;
            const cdx = Math.round(Math.cos(angle));
            const cdy = Math.round(Math.sin(angle));
            if (!isBlocked(a.x + cdx, a.y + cdy)) { dx = cdx; dy = cdy; }
          }
        }

        // Clamp and update position (final obstacle check)
        a.trail.push({ x: a.x, y: a.y });
        if (a.trail.length > 30) a.trail.shift();
        const newX = Math.max(0, Math.min(CFG.gridW - 1, a.x + dx));
        const newY = Math.max(0, Math.min(CFG.gridH - 1, a.y + dy));
        if (!obstacles[newY][newX]) {
          a.x = newX;
          a.y = newY;
        }

        // Update coverage map
        markCovered(a.x, a.y);
      }

      // Detect victims
      for (const v of victims) {
        if (v.found) continue;
        for (const a of agents) {
          const d = Math.sqrt((a.x - v.x) ** 2 + (a.y - v.y) ** 2);
          if (d <= CFG.detectionRadius) {
            v.found = true;
            v.foundStep = step;
            logEvent(`!! VICTIM FOUND at (${v.x},${v.y}) by A${a.id} at t=${step}`, 'log-found');

            // Immediately signal neighbors to Navigate/Form
            a.option = OPTION.FORM;
            a.target = { x: v.x, y: v.y };
            a.optionAge = 0;
            totalSwitches++;
            logEvent(`A${a.id} switched to Form (first responder)`, 'log-form');

            const nbrs = getNeighbors(a.id);
            for (const ni of nbrs) {
              const nb = agents[ni];
              if (nb.option !== OPTION.FORM) {
                nb.option = OPTION.NAVIGATE;
                nb.target = { x: v.x, y: v.y };
                nb.optionAge = 0;
                totalSwitches++;
                logEvent(`A${nb.id} switched to Navigate (graph-conditioned from A${a.id})`, 'log-navigate');
              }
            }
            break;
          }
        }
      }
      // ═══════════════════════════════════════════════════════════════════
      // Simulation step
      // ═══════════════════════════════════════════════════════════════════
      function simStep() {
        if (step >= CFG.maxSteps || victims.filter(v => v.found).length === CFG.nVictims) {
          if (playing) togglePlay();
          return;
        }
        step++;

        // 1) Agent decision & movement
        for (let i = 0; i < CFG.nAgents; i++) {
          const a = agents[i];

          // --- Option Switching Logic (Mock Manager) ---
          a.optionAge++;
          if (a.option !== OPTION.EXPLORE) {
            // If navigating to a target that is already found, revert to explore
            if (a.target) {
              const targetVictim = victims.find(v => v.x === a.target.x && v.y === a.target.y);
              if (targetVictim && targetVictim.found && a.optionAge > 5) {
                a.option = OPTION.EXPLORE;
                a.target = null;
                a.optionAge = 0;
                totalSwitches++;
                logEvent(`A${a.id} cleared target, reverting to Explore`, 'log-explore');
              }
            }
            if (a.optionAge >= CFG.kMax) {
              a.option = OPTION.EXPLORE;
              a.target = null;
              a.optionAge = 0;
              totalSwitches++;
              logEvent(`A${a.id} option timed out, reverted to Explore`, 'log-explore');
            }
          }

          // --- Low-Level Policy Logic (Mock Actor) ---
          let dx = 0, dy = 0;

          if (a.option === OPTION.EXPLORE) {
            // Explore: move toward highest uncertainty (yellow cells) + repel from walls
            let bestScore = -Infinity;
            const moves = [[0, -1], [0, 1], [-1, 0], [1, 0]];

            for (const [ddx, ddy] of moves) {
              const nx = a.x + ddx, ny = a.y + ddy;
              if (nx < 0 || ny < 0 || nx >= CFG.gridW || ny >= CFG.gridH) continue;
              if (obstacles[ny][nx]) continue;

              // Prefer uncovered cells (high belief = unexplored)
              let score = belief[ny][nx] * 3.0;

              // Penalize already-covered cells (coverage overlap penalty)
              score -= coverage[ny][nx] * 0.8;

              // Bonus for being adjacent to obstacles (explore edges of walls)
              for (const [odx, ody] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
                const wx = nx + odx;
                const wy = ny + ody;
                if (wx >= 0 && wy >= 0 && wx < CFG.gridW && wy < CFG.gridH && obstacles[wy][wx]) {
                  score += 0.15;
                  break;
                }
              }

              // Repel from other exploring agents (spread out for max coverage)
              for (const other of agents) {
                if (other.id === a.id) continue;
                if (other.option !== OPTION.EXPLORE) continue;
                const od = Math.sqrt((nx - other.x) ** 2 + (ny - other.y) ** 2);
                if (od < 4) score += od * 0.4;  // reward distance from others
              }

              // Small randomness for tie-breaking
              score += Math.random() * 0.2;

              if (score > bestScore) { bestScore = score; dx = ddx; dy = ddy; }
            }
          } else if (a.option === OPTION.NAVIGATE && a.target) {
            // Navigate toward target victim — obstacle-aware
            const tdx = a.target.x - a.x;
            const tdy = a.target.y - a.y;
            // Try primary direction first; if blocked, try alternate
            let moved = false;
            if (Math.abs(tdx) > Math.abs(tdy)) {
              if (a.x + Math.sign(tdx) >= 0 && a.x + Math.sign(tdx) < CFG.gridW && !obstacles[a.y][a.x + Math.sign(tdx)]) { dx = Math.sign(tdx); moved = true; }
              else if (tdy !== 0 && a.y + Math.sign(tdy) >= 0 && a.y + Math.sign(tdy) < CFG.gridH && !obstacles[a.y + Math.sign(tdy)][a.x]) { dy = Math.sign(tdy); moved = true; }
            } else {
              if (tdy !== 0 && a.y + Math.sign(tdy) >= 0 && a.y + Math.sign(tdy) < CFG.gridH && !obstacles[a.y + Math.sign(tdy)][a.x]) { dy = Math.sign(tdy); moved = true; }
              else if (tdx !== 0 && a.x + Math.sign(tdx) >= 0 && a.x + Math.sign(tdx) < CFG.gridW && !obstacles[a.y][a.x + Math.sign(tdx)]) { dx = Math.sign(tdx); moved = true; }
            }
            // Wall-following fallback
            if (!moved) {
              const alts = [[0, 1], [0, -1], [1, 0], [-1, 0]];
              for (const [adx, ady] of alts) {
                const nx = a.x + adx, ny = a.y + ady;
                if (nx >= 0 && ny >= 0 && nx < CFG.gridW && ny < CFG.gridH && !obstacles[ny][nx]) { dx = adx; dy = ady; break; }
              }
            }
          } else if (a.option === OPTION.FORM && a.target) {
            // Form: circle victim
            const dist = Math.sqrt((a.x - a.target.x) ** 2 + (a.y - a.target.y) ** 2);
            if (dist > 1.5) {
              const tdx = a.target.x - a.x;
              const tdy = a.target.y - a.y;
              if (Math.abs(tdx) >= Math.abs(tdy)) {
                if (a.x + Math.sign(tdx) >= 0 && a.x + Math.sign(tdx) < CFG.gridW && !obstacles[a.y][a.x + Math.sign(tdx)]) dx = Math.sign(tdx);
              } else {
                if (a.y + Math.sign(tdy) >= 0 && a.y + Math.sign(tdy) < CFG.gridH && !obstacles[a.y + Math.sign(tdy)][a.x]) dy = Math.sign(tdy);
              }
            } else {
              const angle = (a.id / CFG.nAgents) * Math.PI * 2 + step * 0.1;
              const cdx = Math.round(Math.cos(angle));
              const cdy = Math.round(Math.sin(angle));
              if (a.x + cdx >= 0 && a.x + cdx < CFG.gridW && a.y + cdy >= 0 && a.y + cdy < CFG.gridH && !obstacles[a.y + cdy][a.x + cdx]) { dx = cdx; dy = cdy; }
            }
          }

          // Clamp and update position
          a.trail.push({ x: a.x, y: a.y });
          if (a.trail.length > 30) a.trail.shift();
          const newX = Math.max(0, Math.min(CFG.gridW - 1, a.x + dx));
          const newY = Math.max(0, Math.min(CFG.gridH - 1, a.y + dy));
          if (!obstacles[newY][newX]) {
            a.x = newX;
            a.y = newY;
          }

          // Update coverage map
          coverage[a.y][a.x]++;
        }

        // 2) Graph updates
        edges = [];
        for (let i = 0; i < agents.length; i++) {
          for (let j = i + 1; j < agents.length; j++) {
            const d = Math.sqrt((agents[i].x - agents[j].x) ** 2 + (agents[i].y - agents[j].y) ** 2);
            if (d <= CFG.proximityRadius) edges.push({ source: i, target: j });
          }
        }

        // 3) Victim detection
        for (const v of victims) {
          if (v.found) continue;
          for (const a of agents) {
            const d = Math.sqrt((a.x - v.x) ** 2 + (a.y - v.y) ** 2);
            if (d <= CFG.detectionRadius) {
              v.found = true;
              v.foundStep = step;
              logEvent(`!! VICTIM FOUND at (${v.x},${v.y}) by A${a.id} at t=${step}`, 'log-found');

              // Immediately signal neighbors to Navigate/Form
              a.option = OPTION.FORM;
              a.target = { x: v.x, y: v.y };
              a.optionAge = 0;
              totalSwitches++;
              logEvent(`A${a.id} switched to Form (first responder)`, 'log-form');

              // Find neighbors in graph
              const nbrs = edges.flatMap(e => e.source === a.id ? e.target : e.target === a.id ? e.source : []);
              for (const ni of nbrs) {
                const nb = agents[ni];
                if (nb.option !== OPTION.FORM) {
                  nb.option = OPTION.NAVIGATE;
                  nb.target = { x: v.x, y: v.y };
                  nb.optionAge = 0;
                  totalSwitches++;
                  logEvent(`A${nb.id} switched to Navigate (graph-conditioned from A${a.id})`, 'log-navigate');
                }
              }
              break;
            }
          }
        }

        // 4) Belief update
        for (let i = 0; i < CFG.nAgents; i++) {
          const bx = agents[i].x, by = agents[i].y;
          belief[by][bx] = 0;
          for (const [dx, dy] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
            const nx = bx + dx, ny = by + dy;
            if (nx >= 0 && nx < CFG.gridW && ny >= 0 && ny < CFG.gridH) belief[ny][nx] *= 0.5;
          }
        }

        updateUI();
        draw();
      }

      // ═══════════════════════════════════════════════════════════════════
      // Drawing
      // ═══════════════════════════════════════════════════════════════════
      function draw(timestamp) {
        animId = requestAnimationFrame(draw);
        if (!ctx) return;

        const speed = parseInt(document.getElementById('speed').value);
        document.getElementById('speedVal').textContent = speed + 'x';

        if (playing) {
          const interval = 1000 / speed;
          if (timestamp - lastFrame >= interval) {
            lastFrame = timestamp;
            simStep();
          }
        }

        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Background
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, W, H);

        // Coverage heatmap (green tint for explored cells)
        for (let y = 0; y < CFG.gridH; y++) {
          for (let x = 0; x < CFG.gridW; x++) {
            const cv = coverage[y][x];
            if (cv > 0) {
              const alpha = Math.min(0.06 + cv * 0.03, 0.22);
              ctx.fillStyle = `rgba(16,185,129,${alpha})`;
              ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
          }
        }

        // Belief heatmap (cyan for unexplored high-prior cells)
        for (let y = 0; y < CFG.gridH; y++) {
          for (let x = 0; x < CFG.gridW; x++) {
            const v = belief[y][x];
            if (v > 0.001) {
              const alpha = Math.min(v * 0.5, 0.35);
              ctx.fillStyle = `rgba(6,182,212,${alpha})`;
              ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
          }
        }
        // Obstacles — dark purple blocks with edge highlights
        for (let y = 0; y < CFG.gridH; y++) {
          for (let x = 0; x < CFG.gridW; x++) {
            if (!obstacles[y][x]) continue;
            const px = x * cellSize, py = y * cellSize;
            // Solid fill
            ctx.fillStyle = '#1a0a2e';
            ctx.fillRect(px, py, cellSize, cellSize);
            // Cross-hatch texture
            ctx.strokeStyle = 'rgba(107,33,168,.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px + cellSize, py + cellSize);
            ctx.moveTo(px + cellSize, py);
            ctx.lineTo(px, py + cellSize);
            ctx.stroke();
            // Edge highlight (top/left light, bottom/right shadow)
            ctx.strokeStyle = 'rgba(139,92,246,.35)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(px, py + cellSize);
            ctx.lineTo(px, py);
            ctx.lineTo(px + cellSize, py);
            ctx.stroke();
            ctx.strokeStyle = 'rgba(0,0,0,.4)';
            ctx.beginPath();
            ctx.moveTo(px + cellSize, py);
            ctx.lineTo(px + cellSize, py + cellSize);
            ctx.lineTo(px, py + cellSize);
            ctx.stroke();
          }
        }

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,.04)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= CFG.gridW; i++) {
          ctx.beginPath();
          ctx.moveTo(i * cellSize, 0); ctx.lineTo(i * cellSize, H);
          ctx.stroke();
        }
        for (let i = 0; i <= CFG.gridH; i++) {
          ctx.beginPath();
          ctx.moveTo(0, i * cellSize); ctx.lineTo(W, i * cellSize);
          ctx.stroke();
        }

        // Graph edges
        for (const [i, j, dist] of edges) {
          const a1 = agents[i], a2 = agents[j];
          const alpha = Math.max(0.08, 0.4 - dist * 0.05);
          ctx.strokeStyle = `rgba(100,116,220,${alpha})`;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo((a1.x + .5) * cellSize, (a1.y + .5) * cellSize);
          ctx.lineTo((a2.x + .5) * cellSize, (a2.y + .5) * cellSize);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Agent trails
        for (const a of agents) {
          if (a.trail.length < 2) continue;
          ctx.strokeStyle = a.color + '30';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo((a.trail[0].x + .5) * cellSize, (a.trail[0].y + .5) * cellSize);
          for (let t = 1; t < a.trail.length; t++) {
            ctx.lineTo((a.trail[t].x + .5) * cellSize, (a.trail[t].y + .5) * cellSize);
          }
          ctx.lineTo((a.x + .5) * cellSize, (a.y + .5) * cellSize);
          ctx.stroke();
        }

        // Victims
        for (const v of victims) {
          const cx = (v.x + .5) * cellSize;
          const cy = (v.y + .5) * cellSize;
          if (v.found) {
            // Found — green checkmark
            ctx.fillStyle = 'rgba(16,185,129,.2)';
            ctx.beginPath(); ctx.arc(cx, cy, cellSize * .6, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.arc(cx, cy, cellSize * .45, 0, Math.PI * 2); ctx.stroke();
            ctx.fillStyle = '#10b981'; ctx.font = `bold ${cellSize * .5}px Inter`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText('V', cx, cy);
          } else {
            // Not found — pulsing red
            const pulse = 0.7 + 0.3 * Math.sin(Date.now() * 0.004);
            ctx.fillStyle = `rgba(244,63,94,${0.12 * pulse})`;
            ctx.beginPath(); ctx.arc(cx, cy, cellSize * .7 * pulse, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = `rgba(244,63,94,${0.5 * pulse})`;
            ctx.beginPath(); ctx.arc(cx, cy, cellSize * .25, 0, Math.PI * 2); ctx.fill();
          }
        }

        // Agents
        for (const a of agents) {
          const cx = (a.x + .5) * cellSize;
          const cy = (a.y + .5) * cellSize;
          const r = cellSize * .38;

          // Glow based on option
          const glow = OPTION_GLOW[a.option];
          ctx.shadowColor = glow;
          ctx.shadowBlur = 14;
          ctx.fillStyle = a.color;
          ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
          ctx.shadowBlur = 0;

          // Inner ring showing option
          const optColor = a.option === 0 ? '#06b6d4' : a.option === 1 ? '#f59e0b' : '#10b981';
          ctx.strokeStyle = optColor;
          ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.arc(cx, cy, r + 3, 0, Math.PI * 2); ctx.stroke();

          // Agent label
          ctx.fillStyle = '#fff';
          ctx.font = `bold ${cellSize * .32}px Inter`;
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(a.id.toString(), cx, cy);

          // Direction indicator for Navigate/Form
          if ((a.option === OPTION.NAVIGATE || a.option === OPTION.FORM) && a.target) {
            const angle = Math.atan2(a.target.y - a.y, a.target.x - a.x);
            const arrowR = r + 10;
            ctx.strokeStyle = optColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx + Math.cos(angle) * (r + 4), cy + Math.sin(angle) * (r + 4));
            ctx.lineTo(cx + Math.cos(angle) * arrowR, cy + Math.sin(angle) * arrowR);
            ctx.stroke();
            // Arrowhead
            ctx.beginPath();
            ctx.arc(cx + Math.cos(angle) * arrowR, cy + Math.sin(angle) * arrowR, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = optColor;
            ctx.fill();
          }
        }

        // Step counter + coverage on canvas
        ctx.fillStyle = 'rgba(255,255,255,.6)';
        ctx.font = '500 13px JetBrains Mono';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`t = ${step}`, 10, 10);
        const cs = getCoverageStats();
        ctx.fillText(`Coverage: ${(cs.pct * 100).toFixed(0)}%`, 10, 28);
        const foundCount = victims.filter(v => v.found).length;
        ctx.textAlign = 'right';
        ctx.fillText(`${foundCount}/${CFG.nVictims} rescued`, W - 10, 10);
      }

      // ═══════════════════════════════════════════════════════════════════
      // UI updates
      // ═══════════════════════════════════════════════════════════════════
      function updateUI() {
        document.getElementById('sStep').textContent = step;
        const found = victims.filter(v => v.found).length;
        document.getElementById('sFound').textContent = `${found} / ${CFG.nVictims}`;
        const H = computeEntropy();
        document.getElementById('sEntropy').textContent = H.toFixed(1);
        document.getElementById('sPotential').textContent = (-H).toFixed(1);
        document.getElementById('sEdges').textContent = edges.length;
        document.getElementById('sSwitches').textContent = totalSwitches;

        // Coverage stats
        const cs = getCoverageStats();
        const covPctStr = (cs.pct * 100).toFixed(1);
        document.getElementById('sCovPct').textContent = covPctStr + '%';
        document.getElementById('sCovBar').style.width = covPctStr + '%';
        document.getElementById('sOverlap').textContent = (cs.overlap * 100).toFixed(1) + '%';

        // Agent list
        const listEl = document.getElementById('agentList');
        listEl.innerHTML = agents.map(a => `
    <div class="agent-row">
      <div class="agent-dot" style="background:${a.color};color:${a.color};"></div>
      <span class="agent-name">A${a.id}</span>
      <span class="agent-option ${OPTION_CLASS[a.option]}">${OPTION_NAME[a.option]}</span>
      <span class="agent-pos">(${a.x},${a.y})</span>
    </div>
  `).join('');
      }

      function logEvent(msg, cls = '') {
        const el = document.getElementById('eventLog');
        el.innerHTML += `<div class="${cls}"><span style="opacity:.5">t=${step}</span> ${msg}</div>`;
        el.scrollTop = el.scrollHeight;
      }

      // ═══════════════════════════════════════════════════════════════════
      // Controls
      // ═══════════════════════════════════════════════════════════════════
      function togglePlay() {
        playing = !playing;
        const btn = document.getElementById('btnPlay');
        btn.textContent = playing ? '⏸ Pause' : '▶ Play';
        btn.classList.toggle('active', playing);
      }

      function stepOnce() {
        if (playing) togglePlay();
        simStep();
      }

      // ── Start ───────────────────────────────────────────────────
      window.addEventListener('DOMContentLoaded', init);
  