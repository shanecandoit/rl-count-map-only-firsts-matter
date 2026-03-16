# Implementation Plan: Epistemic Competition RL

> Phased development from tabular proof-of-concept to publishable Crafter demo.

Each phase builds on the last. Phases 0–4 are the critical path; phases 5–9 are extensions.

---

## Phase 0: Foundations & Scaffolding

**Goal**: Get the repo structure, dependencies, and test harness in place before writing any RL code.

### Tasks

- [ ] `requirements.txt`: torch, gymnasium, minigrid, crafter, numpy, wandb, pytest, matplotlib
- [ ] Directory structure as in README
- [ ] `core/count_map.py`: `GlobalCountMap` class
  - Thread-safe dict mapping `tuple(augmented_state)` → int
  - Methods: `query(s)`, `increment(s)`, `reset_layer(milestone_flags)`, `to_array(shape)`
  - Unit tests: concurrent increment correctness, layer reset, serialization
- [ ] `core/milestone_tracker.py`: `MilestoneTracker` class
  - Tracks which milestone flags are active
  - On flag transition: calls `count_map.reset_layer()`, returns new augmented state
  - Configurable milestone flag list per environment
- [ ] `envs/wrappers.py`: `CompetitiveEnvWrapper`
  - Wraps any Gym env
  - Intercepts `step()`, augments state with milestone flags, computes count-based reward
  - `reward = 1 if count_map.query(aug_state) == 0 else 0`
  - Increments count map after reward computation
- [ ] `analysis/coverage_metrics.py`: `CoverageTracker`
  - Logs: coverage %, milestone hit times, per-agent cell counts
  - Exports to wandb or CSV
- [ ] `tests/`: pytest suite for all core components
- [ ] CI: GitHub Actions running pytest on push

### Success Criteria
- All unit tests pass
- `CompetitiveEnvWrapper` can wrap `gymnasium.make("MiniGrid-Empty-8x8-v0")` and return count-based rewards
- Coverage metrics log correctly for 100 random steps

---

## Phase 1: Tabular Baseline on MiniGrid KeyDoor

**Goal**: Prove the core mechanism works in a fully observable, tabular setting before adding neural networks.

### Environment
- `MiniGrid-DoorKey-5x5-v0` — small enough to enumerate all states
- Milestone flags: `{HasKey: bool, DoorOpen: bool}`
- Augmented state: `(x, y, HasKey, DoorOpen)` — ~100 states per layer, 4 layers max

### Agent
- `agents/tabular_agent.py`: epsilon-greedy Q-table
- Q-table indexed by augmented state
- No neural network — pure tabular update

### Experiments
1. **Single agent**: time to full coverage of all 4 milestone layers
2. **2-agent competitive**: both agents share one `GlobalCountMap`
   - Hypothesis: 2 agents cover the space faster with less redundant visiting
3. **Baseline comparison**: single agent + ε-greedy with NO count rewards (just random exploration)

### Metrics
- Steps to 100% coverage per layer
- Redundant visits (visits to already-claimed states) as % of total steps
- Coverage rate curve (% covered vs. timesteps)

### Visualizations
- Animated heatmap of count map over time, one layer per milestone state
- Side-by-side: single agent vs. 2-agent competitive

### Success Criteria
- 2-agent competitive achieves full coverage in fewer steps than single agent with ε=0.3
- Zero unit test failures
- Coverage heatmap animation renders correctly

---

## Phase 2: Single-Agent Neural Count-Based RL

**Goal**: Replace Q-table with a PPO policy network that takes `[s_t | C_t]` as input. Validate that the neural agent achieves comparable coverage to the tabular baseline.

### Network Architecture
```
Input:  [flat_state | local_count_vector]
        flat_state:         raw obs flattened (or CNN for image obs)
        local_count_vector: n(s_t), n(neighbors_4), n(milestone_neighbors)

FC(256) → ReLU → FC(128) → ReLU → Actor head (softmax) + Critic head (scalar)
```

### Agent
- `agents/ppo_agent.py`: standard PPO (clip ratio 0.2, GAE λ=0.95)
- Count vector appended to observation before first FC layer
- Count values log-normalized: `c_norm = log(1 + n(s)) / log(1 + max_n)`

### Key Design Decisions
- **Non-stationary value target**: because reward drops to 0 once a state is claimed, the value function target is non-stationary. Mitigation: short rollout horizons (128 steps), high discount decay near claimed states.
- **Count normalization**: raw counts grow unboundedly; log-normalize to keep input scale stable.
- **Obs wrapper**: `CountObsWrapper` concatenates count vector to obs before policy sees it.

### Experiments
1. PPO + count input vs. PPO without count input (just intrinsic reward, no count obs)
2. PPO + count input vs. tabular baseline from Phase 1
3. Ablation: does seeing `C_t` in the input improve coverage rate beyond just having count-based reward?

### Success Criteria
- Neural agent matches tabular coverage within 2× sample efficiency
- Adding `C_t` to input improves coverage rate vs. reward-only condition (ablation)

---

## Phase 3: Milestone State Augmentation

**Goal**: Implement the full state augmentation pipeline. Demonstrate that milestone resets create directed re-exploration and faster achievement discovery.

### Environment
- Upgrade to `MiniGrid-DoorKey-8x8-v0`
- Milestone DAG:
  ```
  Start → HasKey → DoorOpen → AtGoal
  ```
- State space grows from 64 cells to 64×8 augmented states (3 binary flags × 64 positions)

### New Components
- `core/milestone_tracker.py` fully implemented:
  - Detects flag transitions from env `info` dict
  - On transition `(HasKey: 0→1)`: marks new layer `(x,y,HasKey=1,DoorOpen=0)` as all-unvisited
  - Emits `milestone_event` for logging
- `MilestoneResetScheduler`: configures which layers to reset on which milestone triggers
  - `PARTIAL_RESET`: only the new layer becomes novel (default)
  - `FULL_RESET`: all layers reset (max re-exploration, not recommended for speed)
  - `LAYER_ONLY`: only states in the new augmented layer become novel (correct behavior)

### Experiments
1. **With milestone augmentation** vs. **without** (flat state space):
   - Does the agent reach `DoorOpen` and `AtGoal` faster with augmentation?
   - Hypothesis: augmentation creates a curriculum — the agent is pulled toward milestones because reaching them opens new novel states
2. **Reward shaping check**: verify the agent isn't exploiting milestone resets by cycling (getting key → dropping key → getting key again). Add `milestone_achieved_once` flag to prevent re-triggering.

### Visualizations
- Layer-by-layer coverage heatmap: one 8×8 grid per augmented state layer
- Milestone discovery timeline: horizontal bar chart of first-hit time per milestone
- "Re-exploration pressure" plot: novel states available vs. timesteps (shows spikes at milestone events)

### Success Criteria
- Agent discovers all 3 milestones within 50k steps on 8×8 grid
- Coverage of `(x,y,HasKey=1)` layer reaches >90% after HasKey milestone hit
- No milestone cycling exploit detected

---

## Phase 4: Competitive Multi-Agent Exploration

**Goal**: Add N agents sharing one `GlobalCountMap`. Validate that the competitive repulsion mechanism produces agent specialization and faster collective coverage.

### Architecture Changes
- `GlobalCountMap` upgraded to multiprocessing-safe shared memory (using `multiprocessing.Manager` or `torch.multiprocessing`)
- Each agent runs in its own process; count map is the only shared state
- Each agent has its own optimizer, replay buffer, policy network
- **No gradient sharing** — only the count map is shared

### Repulsion Mechanism
The repulsion is implicit: agents naturally avoid high-count states because those states yield R=0. No explicit repulsion force needed.

Optional explicit diversity bonus:
```python
R_diversity = λ · (1 - fraction_of_trajectory_also_in_other_agents_recent_buffers)
```

Start without the diversity bonus; add only if agents fail to diverge naturally.

### Experiments
1. **N=1 vs N=2 vs N=4** competitive agents: coverage rate vs. timesteps
2. **Competitive vs. independent** (N agents, each with their OWN count map, no sharing):
   - Hypothesis: shared map + competition beats independent exploration due to repulsion
3. **Policy divergence measurement**:
   - Track per-agent cell visit distributions
   - Compute pairwise KL divergence between agent visit distributions at convergence
   - Hypothesis: divergence is high (agents specialize) vs. near-zero for independent agents

### Metrics
- Collective coverage rate (% of total augmented state space visited by ANY agent)
- Per-agent coverage (breakdown of which agent visited which cells)
- Pairwise KL divergence of visit distributions
- Redundant step ratio (steps where R=0 because another agent already claimed state)

### Visualizations
- Multi-panel heatmap: one panel per agent showing their cells (colored by agent)
- Divergence over time: KL divergence plot as training progresses
- "Territory map": Voronoi-like partition of the grid colored by which agent first visited each cell

### Success Criteria
- 4 competitive agents achieve full coverage 2× faster than 1 agent (total timesteps / N)
- Pairwise KL divergence between agents > 0.3 at convergence (genuine specialization)
- Competitive (shared map) beats independent (separate maps) on collective coverage rate

---

## Phase 5: Teacher-Weighted Count Conditioning

**Goal**: Implement the `W·C_t` input mechanism. A weight vector `W` modulates which regions of the count map the agent prioritizes, enabling interpretable curriculum specification.

### Design

```python
# Policy input construction
count_vector = count_map.get_local_counts(state, radius=3)   # nearby counts
weighted_counts = W * count_vector                            # W is the teacher vector
policy_input = torch.cat([state_embedding, weighted_counts])
```

### Weight Vectors
- `W = [1, 0, 0, 0]`: prioritize layer 0 (base exploration)
- `W = [0, 1, 0, 0]`: prioritize HasKey layer
- `W = [0.5, 0.5, 0, 0]`: balanced between base and HasKey
- `W = uniform`: no curriculum (baseline)

### Experiments
1. **Fixed W vs. uniform W**: does milestone-biased W produce faster milestone discovery?
2. **Heterogeneous agents**: in a 4-agent competitive setting, give each agent a different W (one per milestone layer). Hypothesis: agents specialize by milestone branch.
3. **Interpretability demo**: show that by observing which count dimensions an agent's policy is most sensitive to (via gradient saliency on `W·C_t`), you can infer its "current goal" without access to the reward function.

### Meta-Learner (Optional Extension)
- A meta-controller observes the global count map and dynamically assigns W vectors to agents to maximize collective coverage entropy
- This is curriculum RL via count shaping — a potential independent contribution

### Success Criteria
- Milestone-biased W achieves 25% faster milestone discovery than uniform W
- Gradient saliency on `W·C_t` correctly identifies the agent's milestone priority in >80% of frames

---

## Phase 6: Crafter Main Demo

**Goal**: Scale to Crafter, the primary publishable result. Demonstrate multi-agent competitive exploration discovering Crafter's 22-achievement DAG faster than baselines.

### Why Crafter
- 22 achievements forming a natural dependency DAG (wood → table → pickaxe → stone...)
- Achievement discovery rate is a standard published benchmark
- Small image observations (64×64 or symbolic), manageable compute
- Multiple published baselines to compare against (PPO, Dreamer, Go-Explore)

### Crafter Achievement DAG (milestone structure)
```
collect_wood → make_crafting_table → make_wood_pickaxe → collect_stone
                                   → make_wood_sword
collect_wood → make_crafting_table → make_wood_pickaxe → collect_coal
                                                       → collect_iron
                                                       → make_iron_pickaxe → collect_diamond
...
```

Each achievement is a milestone that expands the augmented state space. The agent gets +1 for visiting `(obs, achievement_flags)` states it hasn't seen.

### State Representation for Crafter
- Raw obs: 9×9 symbolic grid (Crafter's observation mode) — 22 object types
- Achievement flags: 22 binary flags → 2^22 possible layers (too many to enumerate)
- **Practical approximation**: use only the K most recently unlocked flags (K=4)
  - State = (symbolic_obs, last_4_achievement_flags)
  - This keeps the augmented state space tractable while capturing milestone structure

### Count-Based Reward with Pseudo-Counts
Since Crafter obs may not repeat exactly, replace binary registry with:
```
R(s, t) ≈ 1 / sqrt(n_hat(s) + 1)
```
where `n_hat(s)` is a pseudo-count from a density model (CTS or a simple hash).

This is the critical upgrade from Phase 1 (binary) to continuous-space-compatible smooth novelty.

### Baselines to Compare
1. PPO (standard reward)
2. Go-Explore (tabular, from paper)
3. PPO + RND (random network distillation as novelty bonus)
4. **Ours**: N=4 competitive agents + count map + milestone augmentation + W-conditioning

### Experiments
1. **Achievement discovery curve**: # unique achievements unlocked vs. timesteps
2. **Coverage of achievement DAG branches**: do agents specialize by DAG branch?
3. **Ablation**: competitive (shared map) vs. independent vs. no count input
4. **Sample efficiency**: steps to first discover each achievement level (levels 1-5 of DAG depth)

### Compute Requirements
- 4 agents × 8 parallel envs = 32 environment instances
- Expected training: ~10M steps to see clear results
- Approximate: 12-24 hours on a single A100 or 4× V100

### Success Criteria
- Ours beats PPO+RND on achievement discovery rate at 5M steps
- Evidence of DAG branch specialization across agents (different agents discover different branches first)

---

## Phase 7: Continuous State Spaces & Pseudo-Counts

**Goal**: Remove the tabular/discrete-state assumption. Make the system work for pixel observations and continuous control.

### Problem
Binary exact-match registry breaks for continuous spaces (pixel obs never repeat exactly). Two solutions:

**Option A: Hash-based Counts**
```python
# SimHash: locality-sensitive hashing
state_hash = (A @ flatten(obs)) > 0  # A is random Gaussian matrix
count_map[hash] += 1
reward = 1 / sqrt(count_map[hash])
```
Fast, simple, but hash collisions reduce precision.

**Option B: Density Model Pseudo-Counts (recommended)**
```python
# RND-based pseudo-counts
novelty_error = ||fixed_network(s) - learned_network(s)||^2
reward = novelty_error  # high error = novel state
# learned_network trains to predict fixed_network; error drops as state is seen more
```
This is Random Network Distillation (RND) — well-validated, no explicit count map needed.

### Implementation
- `core/pseudo_count.py`: `RNDNoveltyModule`
  - Fixed random network `f: obs → embedding`
  - Learned predictor network `g: obs → embedding`
  - Novelty = MSE(f(obs), g(obs))
  - `g` trains alongside policy; novelty decays as obs become familiar
- Shared RND module across agents: `g` is shared, but each agent contributes gradients
  - This is the continuous-space analog of the shared count map

### Environments
- `CarRacing-v2` (continuous control, pixel obs)
- `Atari/Montezuma's Revenge` (the classic hard-exploration benchmark)

### Experiments
- Does shared RND (competitive) beat per-agent RND on coverage?
- Milestone augmentation with pixel obs: use a learned milestone detector (classify obs → milestone flags)

### Success Criteria
- Competitive shared-RND beats per-agent RND on Montezuma's Revenge score by >20%

---

## Phase 8: World Model + MCTS with Novelty Bias

**Goal**: Add model-based planning. Train a world model `T(s,a) → s'` and use MCTS to plan toward the novelty frontier.

### Why MCTS + Novelty
Without planning, the agent only discovers states by random walk. MCTS can simulate "if I take these 10 actions, I reach a novel state" — dramatically more efficient frontier-seeking.

### World Model
```
T: (obs, action) → next_obs_prediction
   Trained via self-supervised prediction loss
   Architecture: ConvLSTM or Transformer (discrete tokens via VQVAE)
```

Use a lightweight model (not DreamerV3 scale) — goal is planning depth, not pixel-perfect prediction.

### MCTS with Novelty UCT
Replace standard UCT exploitation term:
```
Standard: Q(s,a) + c * sqrt(log N(s) / N(s,a))
Novelty:  Discovery(s') + c * sqrt(log N(s) / N(s,a))

Discovery(s') = 1 / (1 + n(s'))   # high for unvisited states
```

The MCTS tree now guides the agent toward the nearest frontier of unvisited states.

### Cold-Start Problem
The world model is wrong early in training. Mitigation:
1. **Pre-train phase**: run a random policy for 10k steps to seed the replay buffer
2. **Model uncertainty**: add ensemble disagreement as an additional bonus (high disagreement = unreliable model, explore cautiously)
3. **Dyna-style**: alternate between real env steps and model-based planning steps

### Experiments
1. MCTS + novelty bias vs. model-free PPO + count reward: coverage rate comparison
2. Does MCTS find longer paths to novel states? (measure average path length to first novel state per episode)
3. Cold-start ablation: random pre-training vs. no pre-training

### Success Criteria
- MCTS agent finds novel states in 50% fewer steps than model-free agent on MiniGrid-DoorKey-16x16

---

## Phase 9: Analysis, Visualization & Publication Prep

**Goal**: Produce clean figures, ablation tables, and a paper draft.

### Figures to Produce

1. **Main result figure**: Achievement discovery curve on Crafter — ours vs. 3 baselines
2. **Territory map**: 8×8 grid colored by which agent first visited each cell (Phase 4 result)
3. **Milestone discovery timeline**: horizontal bar chart, one bar per achievement per agent
4. **Policy divergence plot**: pairwise KL divergence over training time (shows specialization emerging)
5. **Interpretability demo**: heatmap of gradient saliency on `W·C_t` input dimensions — which counts drive which actions?
6. **Coverage entropy plot**: entropy of visit distribution over time (should plateau → triggers stop condition)

### Ablation Table

| Condition | Achievement Score | Coverage Rate | Policy Divergence |
|-----------|------------------|---------------|-------------------|
| PPO (reward-based) | | | |
| PPO + RND | | | |
| Ours (N=1, no competition) | | | |
| Ours (N=4, independent maps) | | | |
| Ours (N=4, shared map) | | | |
| Ours + W-conditioning | | | |
| Ours + MCTS | | | |

### Paper Structure (target: NeurIPS/ICLR workshop or main track)
1. Introduction: "only the first visit pays" as a design principle
2. Related Work: Go-Explore, NGU, count-based, multi-agent novelty
3. Method: GlobalCountMap, milestone augmentation, competitive dynamics, W-conditioning
4. Experiments: Crafter main result, MiniGrid ablations, policy divergence analysis
5. Analysis: interpretability of W·C_t, emergent specialization, coverage saturation
6. Limitations: hand-crafted milestones, scalability to very large state spaces
7. Conclusion

### Open-Source Release Checklist
- [ ] All experiments reproducible with single command + seed
- [ ] Pre-trained checkpoints uploaded
- [ ] Wandb project public with all runs
- [ ] Colab notebook demo (MiniGrid Phase 4 in-browser)
- [ ] README badges: tests passing, license, arxiv link

---

## Implementation Timeline

```
Week 1-2:   Phase 0 (scaffolding) + Phase 1 (tabular MiniGrid)
Week 3-4:   Phase 2 (PPO + count input) + Phase 3 (milestone augmentation)
Week 5-6:   Phase 4 (competitive multi-agent) — core result
Week 7:     Phase 5 (W-conditioning + interpretability)
Week 8-10:  Phase 6 (Crafter) — requires compute, iterate on results
Week 11:    Phase 7 (continuous spaces) — if Phase 6 looks strong
Week 12:    Phase 8 (MCTS) — stretch goal
Week 13-14: Phase 9 (analysis, writing, release)
```

---

## Known Open Problems (to address during development)

| Problem | Severity | Planned Mitigation |
|---------|----------|--------------------|
| Continuous/high-dim states break binary registry | Critical | Phase 7: pseudo-counts / RND |
| World model cold-start for MCTS | High | Phase 8: random pre-training, Dyna |
| Milestone hand-crafting limits generality | High | Future work: bottleneck state detection |
| Non-stationary value function targets | Medium | Short rollout horizons, frequent resets |
| Competitive dynamics instability | Medium | Start with implicit repulsion only |
| Coverage saturation / endgame | Low | Phase 9: entropy stopping condition |
| MCTS computational cost | Low | Fast learned model + progressive widening |
