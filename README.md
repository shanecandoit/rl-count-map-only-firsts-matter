# Epistemic Competition RL: Only Firsts Matter

> "The reward is zero the second time. Get there first."

A PyTorch research project exploring **competitive novelty-driven exploration** where agents race to discover states first. The reward function is binary and non-stationary: `R=1` for a first visit, `R=0` forever after. Multiple agents share a global count map, creating natural repulsion and specialization.

---

## Core Idea

Standard RL optimizes for cumulative reward. This project optimizes for **state coverage** — the environment model is the product, not the byproduct.

```
R(s, t) = 1   if  n(s) == 0   (first visit globally)
         = 0   otherwise
```

Once a state is claimed by any agent, it yields no reward to anyone. This single rule produces:
- **Exploration pressure**: agents must find new states to earn reward
- **Repulsion between agents**: revisiting another agent's states is wasted effort
- **Natural specialization**: agents diverge into different regions of state space

### State Augmentation via Milestones

When an agent achieves a milestone (e.g., picks up a key), the effective state space expands:

```
Phase A:  state = (x, y)              — 100 cells to discover
Phase B:  state = (x, y, HasKey=1)    — 100 more cells, all novel again
Phase C:  state = (x, y, HasKey=1, DoorOpen=1)  — another 100 cells
```

Each milestone resets novelty for all previously visited positions, giving the agent a fresh exploration horizon with structured goals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Count Map                         │
│          Shared across all agents, read-only per step       │
│          n(s) = visit count for augmented state s           │
│          A state is claimed globally on first visit         │
└────────────────────────┬────────────────────────────────────┘
                         │  n(s_t), n(4 neighbors)  [local_counts]
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Per-Agent Achievement Bit Vector               │
│   bit[i] = 1  →  achievement i already done by this agent  │
│   bit[i] = 0  →  achievement i still novel, reward exists  │
│   Personal to each agent — flips only on that agent's hit   │
└────────────────────────┬────────────────────────────────────┘
                         │  [world_state | achievement_bits | local_counts]
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Policy Network                      │
│   π(a | world_state, achievement_bits, local_counts)        │
│   Trained with bit vector augmentation (p=0.1) so the       │
│   policy generalizes to counterfactual bit patterns         │
└────────────────────────┬────────────────────────────────────┘
                         │  action a_t
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Environment                              │
│   Returns: s_{t+1}, achievement flags, done                 │
│   Reward = 1 if global count map n(aug_state) == 0          │
│          = 0 otherwise  (state already claimed globally)    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `GlobalCountMap` | Thread-safe registry: augmented state → global visit count; shared across all agents |
| `AchievementBitVector` | Per-agent binary flags for each achievement; flips only when that agent personally achieves it |
| `MilestoneTracker` | Detects flag transitions, expands augmented state space, emits milestone events |
| `LocalCountInput` | Extracts visit counts for current cell + 4 neighbors from global map for policy input |
| `CompetitiveEnvWrapper` | Wraps standard Gym envs; computes global-first reward, maintains per-agent bits |
| `CounterfactualProbe` | Inference-time tool: takes world state + modified bit vector, returns action distribution |
| `CoverageMetrics` | Tracks coverage %, milestone hit times, per-agent territory, pairwise KL divergence |

---

## Related Work

| Paper | Overlap |
|-------|---------|
| **Go-Explore** (Ecoffet et al., 2021, *Nature*) | Closest relative — explicit state archive, return-then-explore |
| **Never Give Up / NGU** (Badia et al., 2020) | Episodic + lifelong novelty split; milestone reset ≈ episodic reset with trigger |
| **Count-Based / Pseudo-Counts** (Bellemare et al., 2016) | Core reward mechanism |
| **Multi-Agent NGU** (arXiv 2512.01321, 2025) | Cooperative counterpart; this project is the *competitive* version |
| **Novelty Search** (Lehman & Stanley, 2011) | Same primary objective: novelty over fitness |

**Novel contributions:**
1. Competitive exploration via shared repulsion (vs. cooperative in existing multi-agent novelty work), with explicit global-firsts / per-agent-bits design decision
2. Achievement bit vector as a goal-conditioning + interpretability interface: manipulate bits at inference to probe counterfactuals and redirect behavior without retraining
3. Bit vector augmentation during training (`p_aug=0.1`) to generalize the policy across counterfactual inputs, making the probing interface meaningful
4. Milestone-triggered state augmentation as a clean HRL formalism with provable coverage on tabular environments

---

## Project Phases

See [PLAN.md](PLAN.md) for the full phased implementation plan.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundations & Scaffolding | Planned |
| 1 | Tabular Baseline (MiniGrid) | Planned |
| 1.5 | Crafter Symbolic Bridge (4 achievements, tabular) | Planned |
| 2 | Neural PPO + Bit Vector + Bit Augmentation | Planned |
| 3 | Milestone State Augmentation | Planned |
| 4 | Competitive Multi-Agent (Global Firsts) | Planned |
| 5 | Counterfactual Probing & Human Redirection | Planned |
| 6 | Crafter Main Demo | Planned |
| 7 | Continuous State Spaces | Planned |
| 8 | World Model + MCTS | Planned |
| 9 | Analysis & Publication | Planned |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: tabular baseline on MiniGrid
python train.py --env minigrid-keydoor --agents 1 --phase tabular

# Phase 4: competitive multi-agent
python train.py --env minigrid-keydoor --agents 4 --shared-count-map

# Phase 5: counterfactual probing (Jupyter notebook)
jupyter notebook notebooks/counterfactual_probe.ipynb

# Phase 6: Crafter main demo
python train.py --env crafter --agents 4 --bit-augmentation --log-achievements
```

---

## Metrics

1. **Coverage rate** — % of augmented state space visited vs. timesteps
2. **Milestone discovery time** — steps to first reach each milestone flag
3. **Policy divergence** — pairwise behavioral distance between agents at convergence
4. **Probe accuracy** — % of counterfactual bit vectors where action distribution shifts toward the target achievement's location (interpretability quality metric)
5. **World model generalization** — freeze `T(s,a,s')`, fine-tune on task reward vs. reward-first baseline

---

## Repository Structure

```
rl-count-map-only-firsts-matter/
├── README.md
├── PLAN.md                  # Detailed phased implementation plan
├── requirements.txt
├── train.py                 # Main entry point
├── envs/
│   ├── wrappers.py          # CompetitiveEnvWrapper
│   ├── minigrid_env.py      # MiniGrid setup
│   └── crafter_env.py       # Crafter setup
├── agents/
│   ├── base_agent.py        # Shared agent interface
│   ├── tabular_agent.py     # Phase 1: Q-table baseline
│   ├── ppo_agent.py         # Phase 2+: PPO with count input
│   └── mcts_agent.py        # Phase 8: MCTS with novelty bias
├── core/
│   ├── count_map.py         # GlobalCountMap (thread-safe, shared across agents)
│   ├── milestone_tracker.py # MilestoneTracker + state augmentation
│   ├── bit_vector.py        # AchievementBitVector (per-agent, augmented during training)
│   └── local_counts.py      # LocalCountInput (nearby visit counts for policy)
├── probing/
│   └── counterfactual.py    # CounterfactualProbe — flip bits, compare action distributions
├── notebooks/
│   └── counterfactual_probe.ipynb  # Human-in-the-loop redirection demo
├── experiments/
│   ├── phase1_tabular.py
│   ├── phase1_5_crafter_bridge.py
│   ├── phase4_competitive.py
│   └── phase6_crafter.py
├── analysis/
│   ├── coverage_metrics.py
│   ├── divergence.py
│   └── visualize.py
└── tests/
    ├── test_count_map.py
    ├── test_milestone_tracker.py
    └── test_wrappers.py
```

---

## License

MIT
