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

## Phase 1.5: Crafter Symbolic Bridge (4 Achievements, Tabular)

**Goal**: Bridge the gap between MiniGrid (toy) and full Crafter (complex). Run the same tabular mechanism on Crafter's symbolic observations with only the first 4 achievements unlocked, before adding a neural network.

### Why This Phase Exists
MiniGrid has ~256 augmented states. Full Crafter has 22 achievements and pixel/symbolic obs. Skipping directly to full Crafter makes it hard to debug failures — this phase isolates the environment complexity increase from the neural network complexity increase.

### Environment
- Crafter in symbolic observation mode (9×9 grid, 22 object types per cell)
- Restrict active achievements to: `{collect_wood, make_crafting_table, collect_sapling, collect_coal}`
- These 4 form a shallow sub-DAG, no deep dependencies
- Augmented state: `(symbolic_obs_hash, achievement_bit_vector_4bit)` — 16 layers

### State Hashing
Since symbolic obs are structured arrays (not pixel images), hash them:
```python
state_key = (hash(obs.tobytes()), tuple(achievement_bits))
```
This is exact for symbolic mode (obs are integer grids, no floating point noise).

### Agent
- Same tabular Q-table from Phase 1 but indexed by hashed augmented state
- Counts live in a regular Python dict (state_key → int)

### Experiments
1. Single agent: milestone discovery order and time to first hit each of the 4 achievements
2. 2-agent competitive: does repulsion help in a larger, sparser state space than MiniGrid?
3. Verify the bit vector expands the state space correctly on each achievement unlock

### Success Criteria
- Agent discovers all 4 achievements at least once within 200k steps
- Each achievement unlock visibly spikes "novel states available" in the coverage plot
- State space expansion on achievement unlock verified by unit test

---

## Phase 2: Single-Agent Neural Count-Based RL

**Goal**: Replace Q-table with a PPO policy network that takes `[world_state | achievement_bits | local_counts]` as input. Validate that the neural agent achieves comparable coverage to the tabular baseline.

### Input Structure

The policy receives three distinct inputs, concatenated before the first FC layer:

```
policy_input = [world_state_embedding | achievement_bit_vector | local_counts]

  world_state_embedding : flattened symbolic obs or CNN output for pixels
  achievement_bit_vector: binary flags for each tracked achievement (e.g. 4 or 22 bits)
                          bit[i] = 1 means achievement i already claimed (no reward left there)
                          bit[i] = 0 means achievement i is still novel
  local_counts          : visit counts for the current cell and its 4 neighbors,
                          log-normalized: c_norm = log(1 + n(s)) / log(1 + max_n)
```

The `achievement_bit_vector` is the core signal — it tells the agent what has already been done and therefore where reward still exists. The `local_counts` give spatial novelty awareness without requiring the agent to look up the full global map.

### Network Architecture
```
Input:  [world_state_embedding | achievement_bits | local_counts]

FC(256) → ReLU → FC(128) → ReLU → Actor head (softmax) + Critic head (scalar)
```

### Agent
- `agents/ppo_agent.py`: standard PPO (clip ratio 0.2, GAE λ=0.95)
- Short rollout horizons (128 steps) to mitigate non-stationary value targets

### Key Design Decisions

**Non-stationary value target**: reward drops to 0 once a state is claimed, so the value function target shifts throughout training. Mitigation: short rollouts (128 steps) and frequent target network updates.

**Bit vector augmentation (critical)**: during training, the achievement bit vector always reflects the agent's true current state. At inference, we want to probe counterfactuals — "what would it do if wood was still unclaimed?" But those counterfactual inputs are out-of-distribution if the policy was only trained on DAG-consistent bit vectors.

Fix: randomly corrupt the bit vector during training with probability `p_aug=0.1`:
```python
# During rollout collection only — not used for actual reward computation
if random.random() < p_aug:
    bits_for_policy = randomly_flip_some_bits(true_bits, flip_prob=0.15)
else:
    bits_for_policy = true_bits
```
This forces the policy to generalize across counterfactual bit patterns, making the inference-time probing meaningful.

### Experiments
1. PPO + `[world_state | bits | local_counts]` vs. PPO without count/bit input
2. PPO + full input vs. tabular baseline from Phase 1
3. **Augmentation ablation**: with vs. without bit vector augmentation (`p_aug=0` vs `p_aug=0.1`) — does augmentation improve counterfactual probe quality without hurting training performance?

### Success Criteria
- Neural agent matches tabular coverage within 2× sample efficiency
- Adding achievement bits + local counts improves coverage rate vs. reward-only (ablation)
- With `p_aug=0.1`: setting a bit to 0 at inference visibly changes the action distribution toward that achievement's region

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

### Explicit Design Decision: Global Firsts vs. Per-Agent Firsts

This is the most important design choice in the whole system. Two options:

**Option A — Global Firsts (competitive)**
The count map is shared. If Agent A visits `(x=5, y=5, HasKey=1)`, that state is claimed. Agent B earns R=0 for that state forever. The competitive pressure is real and strong.
- Pro: genuine racing dynamic, agents must diverge to earn reward
- Con: Agent B's achievement bit vector never flips for wood (Agent A picked it up, not B)
- Con: late-joining agents get squeezed out of rewarding states

**Option B — Per-Agent Firsts (cooperative count, competitive credit)**
Each agent has its own novelty registry. The *global* count map is only used for `local_counts` (spatial awareness), not for reward. Each agent's reward is `1` the first time *it* visits a state, regardless of other agents.
- Pro: all agents always have reward signal; no agent gets starved
- Pro: the bit vector stays meaningful per-agent (reflects that agent's achievements)
- Con: weaker competitive pressure; agents may converge on same behavior

**Decision: use Option A (global firsts) with per-agent achievement bit vectors.**

The count map is global — spatial states are claimed globally. But each agent's *achievement bit vector* is local — it only flips when that agent personally achieves the milestone. This means:
- Two agents can both earn reward for visiting `(x=5, y=5)` in the `HasKey=1` layer *if they each got the key themselves*
- But only the first agent to visit `(x=5, y=5, HasKey=1)` globally earns reward for that augmented state
- The bit vector is the agent's personal achievement history; the count map is shared world knowledge

This preserves both the competitive dynamic (global map) and the interpretability interface (per-agent bits).

### Architecture Changes
- `GlobalCountMap` upgraded to multiprocessing-safe shared memory (using `multiprocessing.Manager` or `torch.multiprocessing`)
- Each agent runs in its own process; count map is the only shared state
- Each agent has its own: optimizer, replay buffer, policy network, achievement bit vector
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

## Phase 5: Bit Vector Goal Conditioning & Counterfactual Probing

**Goal**: Validate the bit vector as an interpretable goal-conditioning and human-interactive interface. Demonstrate that manipulating the bit vector at inference time produces predictable, meaningful behavioral changes — without any additional machinery.

### The Core Idea

The achievement bit vector is simultaneously:
1. **A reward signal** during training: `bit[i]=0` means achievement i is unclaimed, reward is available
2. **A goal specification** at inference: set `bit[wood]=0` to tell the agent "treat wood as ungathered, go get it"
3. **A counterfactual interface** for humans: manipulate bits to explore "what would it do if..."

No teacher, no weight vector, no saliency maps required. The interpretability is by construction — the agent was trained to respond to bit states, so probing with modified bits is meaningful as long as augmentation (Phase 2) generalized the policy across counterfactual inputs.

### The Counterfactual Probe Protocol

```python
def probe_policy(policy, world_state, true_bits, counterfactual_bits):
    """
    Take a fixed world_state, compare action distributions under
    true vs. counterfactual bit vectors.
    """
    with torch.no_grad():
        dist_true   = policy(world_state, true_bits,          local_counts)
        dist_counter = policy(world_state, counterfactual_bits, local_counts)
    return dist_true, dist_counter, kl_divergence(dist_true, dist_counter)
```

**Example probes on MiniGrid KeyDoor (8×8):**

| Probe | True bits | Counterfactual bits | Expected behavior change |
|-------|-----------|--------------------|-----------------------------|
| "Pretend no key" | `[HasKey=1, DoorOpen=0]` | `[HasKey=0, DoorOpen=0]` | Agent should move toward key location |
| "Pretend door open" | `[HasKey=1, DoorOpen=0]` | `[HasKey=1, DoorOpen=1]` | Agent should move toward goal |
| "Pretend nothing done" | `[HasKey=1, DoorOpen=1]` | `[0, 0]` | Agent returns to exploration mode |

### Human-in-the-Loop Interface

The bit vector is human-legible by design. A person watching the agent can:
- Read the bit vector and know exactly what the agent "thinks it still needs to do"
- Override individual bits to redirect behavior without retraining
- Use the probe protocol to understand *why* the agent is doing what it's doing

This is different from post-hoc interpretability (saliency, probes, LIME). The bit vector is the goal representation; manipulating it is the interpretability mechanism.

### Experiments

1. **Goal-directed bit setting**: freeze a trained policy; set `bit[target_achievement]=0`, all others=1. Measure:
   - Does the agent navigate toward the target achievement?
   - How does KL divergence between `true_bits` and `counterfactual_bits` action distributions correlate with how "surprising" the counterfactual is (e.g., DAG-impossible vs. DAG-consistent)?

2. **Augmentation quality check**: compare policies trained with `p_aug=0` vs `p_aug=0.1` on counterfactual probe quality. Metric: does the counterfactual action distribution shift in the *right direction* (toward the target achievement) vs. random noise?

3. **Human redirection demo** (qualitative): in a live MiniGrid session, human observer flips bits in real time and records whether the agent noticeably changes direction. This is the interpretability paper figure.

4. **Out-of-distribution boundary**: systematically test DAG-impossible bit vectors (e.g., `HasIronPickaxe=1, HasCraftingTable=0`). Measure entropy of action distribution — high entropy = policy is confused, low entropy = policy has generalized. Ideally entropy is moderate and the behavior is plausible.

### What We Are Not Doing

- No W weight vector ("teacher" prioritization) — the bit vector already encodes what's done/undone, that's sufficient
- No gradient saliency on the count input — probing is direct, not indirect
- No meta-learner for dynamic W assignment — out of scope

### Success Criteria
- On MiniGrid: setting `bit[HasKey]=0` at inference reliably shifts action distribution toward key location in >80% of sampled world states
- Augmented policy (`p_aug=0.1`) produces lower entropy and more directed behavior on counterfactual probes vs. non-augmented policy
- Human redirection demo works end-to-end in a Jupyter notebook

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
4. **Ours**: N=4 competitive agents + global count map + per-agent achievement bits + bit vector augmentation

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
5. **Counterfactual probe figure** (Phase 5 result): 3-panel figure showing a fixed world state, the true bit vector action distribution, and two counterfactual action distributions — arrows on the grid showing where each version of the agent moves
6. **Coverage entropy plot**: entropy of visit distribution over time (should plateau → triggers stop condition)
7. **Human redirection demo**: screenshot sequence from the Jupyter notebook showing bit flips changing agent trajectory in real time

### Ablation Table

| Condition | Achievement Score | Coverage Rate | Policy Divergence | Probe Accuracy |
|-----------|------------------|---------------|-------------------|----------------|
| PPO (reward-based) | | | | n/a |
| PPO + RND | | | | n/a |
| Ours (N=1, no competition) | | | | |
| Ours (N=4, independent maps) | | | | |
| Ours (N=4, shared map, global firsts) | | | | |
| Ours + bit vector augmentation (p=0.1) | | | | |
| Ours + MCTS | | | | |

*Probe accuracy*: % of counterfactual probes where action distribution shifts toward the target achievement's location.

### Paper Structure (target: NeurIPS/ICLR workshop or main track)
1. Introduction: "only the first visit pays" as a design principle; bit vector as goal + interpretability interface
2. Related Work: Go-Explore, NGU, count-based, GCRL/UVFA, multi-agent novelty
3. Method: GlobalCountMap, per-agent achievement bit vector, global firsts decision, bit vector augmentation, counterfactual probe protocol
4. Experiments: Crafter main result, MiniGrid ablations, policy divergence analysis
5. Analysis: counterfactual probe quality, human redirection demo, coverage saturation
6. Limitations: hand-crafted milestones, DAG-impossible probes are out-of-distribution, scalability
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
Week 3:     Phase 1.5 (Crafter symbolic bridge, 4 achievements, tabular)
Week 4-5:   Phase 2 (PPO + bit vector + local counts + augmentation)
Week 6:     Phase 3 (milestone state augmentation, full DAG)
Week 7-8:   Phase 4 (competitive multi-agent, global firsts) — core result
Week 9:     Phase 5 (counterfactual probing, human redirection demo)
Week 10-12: Phase 6 (Crafter full demo) — requires compute, iterate
Week 13:    Phase 7 (continuous spaces) — if Phase 6 looks strong
Week 14:    Phase 8 (MCTS) — stretch goal
Week 15-16: Phase 9 (analysis, writing, release)
```

---

## Known Open Problems (to address during development)

| Problem | Severity | Planned Mitigation |
|---------|----------|--------------------|
| Continuous/high-dim states break binary registry | Critical | Phase 7: pseudo-counts / RND |
| Counterfactual bit vectors are out-of-distribution | High | Phase 2: bit vector augmentation (p_aug=0.1) |
| World model cold-start for MCTS | High | Phase 8: random pre-training, Dyna |
| Milestone hand-crafting limits generality | High | Future work: bottleneck state detection |
| Non-stationary value function targets | Medium | Short rollout horizons, frequent target resets |
| Competitive dynamics instability | Medium | Start with implicit repulsion (global firsts) only |
| DAG-impossible counterfactuals confuse policy | Medium | Measure probe entropy; only present DAG-consistent probes to humans |
| Coverage saturation / endgame | Low | Phase 9: entropy stopping condition |
| MCTS computational cost | Low | Fast learned model + progressive widening |
