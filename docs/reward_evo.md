# Architectural Manifesto: Physics & Math Justification

Before detailing the evolutionary stages of the agent's behavior, it is crucial to establish that the underlying physics engine, reward functions, and spatial geometry are mathematically sound and physically realistic. The training bottlenecks encountered were cognitive (agent capacity) rather than environmental.

## 1. Thrust-to-Weight Ratio and Hover Bias
- **Drone Mass:** $1.0 \text{ kg}$
- **Gravity:** $9.81 \text{ m/s}^2$
- **Hover Force Required:** $2.4525 \text{ N}$ per motor.

The control space maps the agent's continuous output $[-1.0, 1.0]$ to motor thrust using the formula:
$$F = 2.45 + (\text{action} \times 5.0)$$

This means an output of $0.0$ perfectly counteracts gravity, creating a natural **Hover Bias**. Furthermore, the maximum possible thrust per motor is $7.45 \text{ N}$, resulting in a total upward force of $\approx 29.8 \text{ N}$. This gives the drone a **3:1 Thrust-to-Weight Ratio**, which perfectly mirrors the agile flight dynamics of real-world racing quadcopters. The forces are clamped between $[0.0, 10.0]$ to strictly prevent physically impossible negative thrust.

## 2. The Curriculum Initialization Distance ("The Golden Ratio")
In Stage 0, the first coin is spawned exactly $1.0 \text{ meter}$ in front of the drone. This is not an arbitrary number; it mathematically balances the reward economy to prevent the agent from bleeding points simply for existing.

Based on our YAML config:
- `alive_bonus`: $0.02$
- `distance_penalty_multiplier`: $0.02$

The net reward at spawn is calculated as:
$$\text{Net Reward} = \text{Alive Bonus} - (\text{Distance Penalty} \times \text{Distance})$$
$$\text{Net Reward} = 0.02 - (0.02 \times 1.0) = \mathbf{0.0}$$

At a distance of exactly $1.0 \text{ m}$, the agent receives exactly $0.0$ points per step. It does not suffer penalty bleed, nor does it farm free points. If the agent moves even $1 \text{ cm}$ closer (distance $= 0.99 \text{ m}$), the equation yields a positive reward, instantly teaching the agent that forward movement equals profit.

## 3. Cognitive Upgrades (Resolving the Training Bottleneck)
Despite a mathematically perfect environment, the initial `[64, 64]` Multilayer Perceptron (MLP) architecture failed to learn stable flight. 

- **Brain Capacity:** Controlling 4 independent rotors in 3D space based on a 32-D observation vector (Kinematics + 16-ray Lidar) requires significant cross-correlation capabilities. The network was upgraded to an industry-standard `[256, 256]` architecture to provide the necessary capacity for complex aerodynamic modeling (such as pitch braking without overshoot).
- **High-Frequency Control (240Hz):** Real-world quadcopters require extremely high-frequency PID loops (often 400Hz+) to maintain stability. The agent directly controls the rotors at the simulation frequency of 240Hz, allowing for micro-corrections and preventing unrecoverable aerodynamic flips.


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
**Behavior:** With the suicide bug fixed, the agent realized that the `Alive Bonus` was so high, and the `Distance Penalty` so low, that it didn't need to hunt for coins. It just flew straight up to the ceiling (where there are no obstacles) and crushed to the ceiling (as it couldn't learn to safely hover yet).
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
```

## Stage 0.1: Kamikaze Policy (Economic Reform)
**Behavior:** The agent successfully solved the Sparse Reward problem and grabbed the first coin (+300). However, it refused to learn how to brake or stabilize. It discovered a mathematical loophole: diving aggressively into the coin and immediately crashing (-50) yielded a massive net profit of +250. It optimized for a quick death rather than sustainable flight.
**Fix:** Restructured the reward economy in the YAML configuration. Increased distance and velocity penalties to discourage erratic, high-speed dives. Crucially, matched the `collision_penalty` (300) to the `coin_collection_reward` (300). Now, crashing immediately after collecting a coin results in a net-zero sum, forcing the agent to learn deceleration and hover stabilization to preserve its profits.

**Code Changes:**
```yaml
# CHANGED: configs/teacher_ppo.yaml
# OLD:
  distance_penalty_multiplier: 0.001
  velocity_penalty_multiplier: 0.001
  collision_penalty: 50.0

# NEW: Enforcing sustainable flight economics
  distance_penalty_multiplier: 0.02
  velocity_penalty_multiplier: 0.01
  collision_penalty: 300.0  # Matched to coin reward to kill the Kamikaze profit
```

## Stage 0.2: The Local Optima Trap & Hovercraft Mode (Task Decomposition)
**Behavior:** After 4 Million steps, the agent's entropy dropped from -6 to -1, and its reward flatlined perfectly at -50. It learned that hitting the first coin (+300) and immediately losing control due to coupled dynamics (pitching forward causes altitude loss, correcting altitude causes upward rocket momentum) and crashing into the ceiling (-300) was the safest bet. It refused to explore further.
**Fix:** Implemented **"Hovercraft Mode" (Z-Axis Lock)** for Stage 0. By dynamically locking the drone's Z-coordinate to 2.0 meters programmatically, we removed gravity and altitude control from the agent's cognitive load. The agent can now safely pitch, roll, and yaw to translate across the X/Y plane without catastrophic Z-axis momentum spikes. This allows it to learn targeting and braking in 2D before unlocking full 3D physics in Stage 1.

**Code Changes:**
```yaml
# ADDED in YAML: lock_z flag dynamically controls Hovercraft Mode
  stage_0:
    lock_z: True # 2D Hovercraft
  stage_1:
    lock_z: False # 3D Full Flight
```

```python
# ADDED in drone_sim.py step(): Programmatic Z-Lock (Hovercraft Mode)
if self.lock_z:
    pos, ori = p.getBasePositionAndOrientation(self.drone_id)
    lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
    # Reset position to Z=2.0, kill Z velocity
    p.resetBasePositionAndOrientation(self.drone_id, [pos[0], pos[1], 2.0], ori)
    p.resetBaseVelocity(self.drone_id, [lin_vel[0], lin_vel[1], 0.0], ang_vel)
```

## Stage 0.3: Wall-Hugging & Physics Exploitation (The "Hovercraft" Loophole)
**Behavior:** The implementation of Hovercraft Mode (Z-Lock) successfully stopped the vertical Kamikaze behavior. However, the agent quickly found a new local optima: "Physics Exploitation." It realized that maintaining aerodynamic balance in free space requires high entropy and continuous micro-corrections, but sliding against a wall provides infinite physical stability. It flew straight to the room's boundary and leaned its rigid body against the wall, perfectly stabilizing its X/Y axes and farming the alive bonus without triggering the 75-degree tilt penalty. Because PyBullet lacks a "propeller destruction" mechanic, the agent treated the wall as a safety rail.
**Fix:** We bridged the "Sim2Real" gap by treating the mathematical room boundaries as highly lethal physical obstacles. We appended the generated `wall_ids` to the physical collision detection loop. Now, even a millimeter of contact with a wall results in immediate termination and a severe `-300` collision penalty. This strict constraint forcefully evicts the agent from its "contact-rich" comfort zone, compelling it to learn true "free-flight" stabilization in the center of the room.

**Code Changes:**
```python
# CHANGED in drone_sim.py step(): Added wall collision detection
# OLD:
# for obs_id in self.obstacle_ids:
#     contact_points = p.getContactPoints(bodyA=self.drone_id, bodyB=obs_id)

# NEW: Bridging Sim2Real by electrifying the walls
for entity_id in self.obstacle_ids + self.wall_ids:
    contact_points = p.getContactPoints(bodyA=self.drone_id, bodyB=entity_id)
    if len(contact_points) > 0:
        is_collision = True
        terminated = True
        info['is_success'] = False
        break
```

## Stage 0.4: Observation Space & Physics Fix (Quaternion → Euler + Yaw Torque)

**Behavior:** Despite Hovercraft Mode (Z-Lock) preventing vertical crashes, the agent continued 
to exhibit uncontrolled spinning and failed to learn stable directional flight. 
Two root-cause bugs were identified in the environment itself, independent of reward shaping.

**Bug 1 — Quaternion Observation:**
The orientation was passed to the agent as a raw quaternion (4 values: x, y, z, w). 
Quaternion space is non-linear and discontinuous — a network cannot learn that 
[0,0,0,1] and [0,0,1,0] represent a 90-degree rotation difference. 
The agent received mathematically uninterpretable orientation data.

**Bug 2 — Missing Yaw Torque:**
All 4 motors applied only upward force `[0, 0, F]` with no counter-rotating torque. 
In a real quadcopter, motors 0 & 2 spin clockwise and motors 1 & 3 spin counter-clockwise, 
canceling yaw torque. Without this, the drone freely spins around the Z-axis 
and the agent wastes its entire policy capacity trying to fight an uncontrollable rotation.

**Fix:**
1. **Euler Angles:** Replaced quaternion with Euler angles (roll, pitch, yaw) in the 
   observation vector. Observation space reduced from 32-D to 31-D.
2. **Yaw Torque:** Added differential torque based on motor force imbalance:
   `torque = ((F0 + F2) - (F1 + F3)) * 0.01`
   This gives the agent a physically realistic mechanism to control yaw.

**Code Changes:**
```python
# CHANGED: observation_space shape 32 -> 31
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

# CHANGED: _get_obs() quaternion -> euler
euler = p.getEulerFromQuaternion(ori)
obs = np.concatenate([drone_pos, euler, linear_vel, angular_vel, relative_pos, lidar_data])

# ADDED: step() yaw torque after force application
torque_mag = ((forces[0] + forces[2]) - (forces[1] + forces[3])) * 0.01
p.applyExternalTorque(self.drone_id, -1, [0, 0, torque_mag], flags=p.LINK_FRAME)
```

## Stage 0.5: The Hierarchical Paradigm Shift (Abandoning Raw Thrust)

**Behavior:** Despite all environmental constraints, bug fixes, and reward shaping, the agent's entropy consistently collapsed from -6 to -1 around the 2 Million step mark. Episode lengths rarely exceeded 300-400 steps (approx. 1.5 seconds at the 240Hz simulation rate). 
The root cause was cognitive overload: the agent was tasked with simultaneously learning non-linear aerodynamics, rigid body stabilization, high-frequency motor mixing, and 3D spatial navigation. Because learning basic flight physics from scratch via random exploration is nearly impossible, the agent resorted to immediate termination (crashing) to minimize ongoing penalties.

**Fix:** Implemented **Hierarchical Control**, moving away from the "End-to-End" approach of directly controlling the 4 motor thrusts.
1. **High-Level (PPO Agent):** The agent's 4D action space no longer outputs raw motor forces. Instead, it is mapped to intuitive flight commands: `[Target Pitch, Target Roll, Target Yaw Rate, Target Thrust]`.
2. **Low-Level (PD Controller):** A custom Proportional-Derivative (PD) controller was integrated directly into the environment's `step()` function. It reads the agent's target attitude, calculates the current error, generates the necessary correcting torques, and performs X-configuration motor mixing.

This architectural shift abstracts away the high-frequency physical stabilization, allowing the RL agent to focus 100% of its policy capacity on its actual goal: obstacle avoidance and optimal pathfinding.

**Code Changes:**
```python
# CHANGED in drone_sim.py: Action Space Mapping
# OLD: actions directly mapped to individual motor forces [F0, F1, F2, F3]
# NEW: actions map to high-level attitude targets
target_pitch = action[0] * (math.pi / 6)
target_roll = action[1] * (math.pi / 6)
target_yaw_rate = action[2] * 2.0
target_thrust = ((action[3] + 1.0) / 2.0) * 20.0

# ADDED in drone_sim.py: Low-Level PD Controller & Motor Mixer
tau_pitch = (Kp_ang * pitch_error) - (Kd_ang * ang_vel[1])
tau_roll = (Kp_ang * roll_error) - (Kd_ang * ang_vel[0])
tau_yaw = Kp_yaw * yaw_rate_error

# X-Configuration Motor Mixing
f0 = base_f - tau_pitch + tau_roll - tau_yaw
f1 = base_f + tau_pitch + tau_roll + tau_yaw
f2 = base_f + tau_pitch - tau_roll - tau_yaw
f3 = base_f - tau_pitch - tau_roll + tau_yaw
```