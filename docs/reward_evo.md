# Evolution of the Drone Reward Function & Constraint Shaping

This document logs the chronological evolution of our RL agent's behavior, detailing the "Reward Hacking" (local optima) bugs encountered and the mathematical/physical fixes applied to reach the final `Hunter Model` policy.

## Phase 0: Playing Possum (Lazy Local Optima)
**Behavior:** The drone realized flying was risky. It would immediately turn off its motors, fall to the floor, and simply lie flat to collect the continuous `Alive Bonus` without taking any collision penalties.
**Fix:** We introduced a strict Z-axis constraint. If the center of mass drops too low, it's considered a fatal collision.

**Code Changes:**
```python
# ADDED physical constraint
if (abs(drone_pos[0]) > hx - 0.2 or 
    abs(drone_pos[1]) > hy - 0.2 or 
    # FIX: Agent dies if it rests on the floor
    drone_pos[2] < 0.1 or  
    drone_pos[2] > hz * 2 - 0.2):
    is_collision = True
    terminated = True
    info['is_success'] = False
```

## Phase 1: The Breakdance (Effort Hacking)
**Behavior:** Forced to stay above `Z=0.1`, the agent tried to hover just slightly off the ground, often flipping over and wildly spinning its rotors against the floor to maintain height without flying properly.
**Fix:** We raised the floor death limit, added a severe penalty for tilting (Euler angles), and penalized motor effort to discourage chaotic spinning.

**Code Changes:**
```python
# CHANGED Z-limit
# From: drone_pos[2] < 0.1
# To: drone_pos[2] < 0.3

# ADDED Tilt (Euler) Penalty
if abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
    is_collision = True
    terminated = True
    info['is_success'] = False

# ADDED Effort Penalty
effort = np.sum(np.square(action))
reward -= 0.001 * effort
```

## Phase 2: Suicide Policy (Distance Penalty > Alive Bonus)
**Behavior:** Now forced to fly smoothly and upright, the agent faced a new problem: the penalty for being far from the coin was too massive. It mathematically calculated that staying alive was a net negative, so it deliberately crashed into walls to end the episode quickly.
**Fix:** Reduced the distance penalty drastically so the agent could survive long enough to explore without bleeding points.

**Code Changes:**
```python
# CHANGED Distance Penalty
# OLD: reward -= 0.05 * current_distance
# NEW: reward -= 0.005 * current_distance 
```

## Phase 3: Ceiling Hugging (Icarus Effect)
**Behavior:** With the suicide bug fixed, the agent realized that the `Alive Bonus` was so high, and the `Distance Penalty` so low, that it didn't need to hunt for coins. It just flew straight up to the ceiling (where there are no obstacles) and safely hovered there to farm points.
**Fix:** Implemented the final **Hunter Model**. Drastically reduced the alive bonus, tied it mathematically to the distance penalty (canceling each other out at long ranges), and significantly increased the coin reward.

**Code Changes:**
```python
# CHANGED Alive Bonus (Reduced by 10x)
# OLD: reward += 0.2
# NEW: reward += 0.02

# CHANGED Distance Penalty (Fine-tuned to balance)
# OLD: reward -= 0.005 * current_distance
# NEW: reward -= 0.001 * current_distance

# CHANGED Coin Reward (Massive incentive to hunt)
# OLD: reward += 100.0
# NEW: reward += 300.0
```




# Curriculum Tuning Log (Hyperparameters)
## Stage 0: The Kamikaze Dive & Sparse Reward (Action Mapping & Curriculum Init)
**Behavior:** Even with perfectly tuned dense rewards, the agent failed to learn and showed a flatline learning curve. It suffered from two critical issues: 
1. The raw action output of `0.0` resulted in `0 N` thrust, causing the drone to drop like a rock. Negative actions caused physically impossible reverse thrust, pulling the drone into an immediate 75-degree fatal tilt.
2. In a $2048 \text{ m}^3$ room, the probability of randomly stumbling into a $40 \text{ cm}$ reward zone was near zero (Sparse Reward problem), meaning the agent never discovered the massive `+300` coin reward.
**Fix:** 1. **Hover Bias (Action Normalization):** Mapped the agent's $[-1.0, 1.0]$ output space so that $0.0$ equals the exact physical hover force ($2.45 \text{ N}$ per motor), and clamped forces to $[0.0, 10.0] \text{ N}$ to enforce physical limits.
2. **The "Force-Feeding" Trick (Curriculum Initialization):** For Stage 0, hardcoded the first coin to spawn exactly $1.0 \text{ m}$ in front of the drone's nose to guarantee early reward discovery and break the local optima of "just staying alive".

**Code Changes:**
```python
# CHANGED: Action Space Mapping (Hover Bias & Reverse Thrust Fix)
# OLD: forces = np.clip(action, -1.0, 1.0) * 10.0
# NEW:
hover_force = 9.81 / 4.0 
raw_forces = hover_force + (action * 5.0)
forces = np.clip(raw_forces, 0.0, 10.0)

# CHANGED: Coin Spawning (Curriculum Stage 0)
# OLD: All 4 coins were placed exactly in the far corners.
# fixed_positions = [[4.0, 4.0, 2.0], [-4.0, 4.0, 2.0], [4.0, -4.0, 2.0], [-4.0, -4.0, 2.0]]
# NEW: Spawning the first target directly in the flight path (1 meter ahead) to teach the reward.
fixed_positions = [
    [1.0, 0.0, 2.0],  # Right in front of the drone's nose!
    [0.0, 1.5, 2.0],
    [4.0, 4.0, 2.0],
    [-4.0, -4.0, 2.0]
]