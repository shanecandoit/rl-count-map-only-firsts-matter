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
└──────────────┬──────────────────────────────────────────────┘
               │  n(s_t), n(neighbors)
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Policy Network                      │
│   Input:  [s_t | W · C_t]                                  │
│     s_t   = current raw state                               │
│     C_t   = local count vector (visit counts near s_t)      │
│     W     = teacher weight vector (milestone priorities)    │
│   Output: action distribution π(a | s_t, C_t, W)           │
└──────────────┬──────────────────────────────────────────────┘
               │  action a_t
               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Environment                              │
│   Returns: s_{t+1}, milestone flags, done                   │
│   Reward computed externally from count map                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `GlobalCountMap` | Thread-safe registry mapping augmented states → visit count |
| `MilestoneTracker` | Detects flag transitions, triggers state space expansion |
| `WeightedCountInput` | Constructs `W·C_t` input vector for policy conditioning |
| `CompetitiveEnvWrapper` | Wraps standard Gym envs, injects count-based reward |
| `CoverageMetrics` | Tracks coverage rate, milestone discovery time, agent divergence |

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
1. Competitive exploration via shared repulsion (vs. cooperative in existing multi-agent novelty work)
2. Teacher-weighted count conditioning `W·C_t` as interpretable curriculum mechanism
3. Milestone-triggered state augmentation as a clean HRL formalism

---

## Project Phases

See [PLAN.md](PLAN.md) for the full phased implementation plan.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundations & Scaffolding | Planned |
| 1 | Tabular Baseline (MiniGrid) | Planned |
| 2 | Single-Agent Count-Based RL | Planned |
| 3 | Milestone State Augmentation | Planned |
| 4 | Competitive Multi-Agent | Planned |
| 5 | Teacher-Weighted Curriculum | Planned |
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

# Phase 6: Crafter main demo
python train.py --env crafter --agents 4 --weighted-counts --log-achievements
```

---

## Metrics

1. **Coverage rate** — % of augmented state space visited vs. timesteps
2. **Milestone discovery time** — steps to first reach each milestone flag
3. **Policy divergence** — pairwise behavioral distance between agents at convergence
4. **World model generalization** — freeze `T(s,a,s')`, fine-tune on task reward vs. reward-first baseline

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
│   ├── count_map.py         # GlobalCountMap (thread-safe)
│   ├── milestone_tracker.py # MilestoneTracker + state augmentation
│   └── weighted_counts.py   # W·C_t construction
├── experiments/
│   ├── phase1_tabular.py
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
