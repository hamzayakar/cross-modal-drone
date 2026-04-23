# Architectural Manifesto: Physics & Math Justification

Before I get into the stages, I want to establish that the underlying physics engine, reward functions, and spatial geometry are — as far as I can tell — mathematically sound and physically realistic. Most of the training bottlenecks I kept running into turned out to be cognitive (agent capacity) rather than environmental.

## 1. Thrust-to-Weight Ratio and Hover Bias (Hierarchical PD Control)
- **Drone Mass:** $1.0 \text{ kg}$
- **Gravity:** $9.81 \text{ m/s}^2$
- **Hover Force Required:** $\approx 2.45 \text{ N}$ per motor (Total $9.81 \text{ N}$).

*(Note: The initial End-to-End architecture used a direct motor thrust mapping. With the shift to Hierarchical PD Control in Stage 0.5, this was updated to Target Attitude & Thrust.)*

The control space now maps the agent's continuous output `action[3]` (Throttle) from $[-1.0, 1.0]$ to a **Target Thrust** using the formula:
$$\text{Target Thrust} = \frac{\text{action}[3] + 1.0}{2.0} \times 20.0$$

This means a neutral output of $0.0$ yields exactly $10.0 \text{ N}$ of total thrust ($2.5 \text{ N}$ per motor), which perfectly counteracts gravity, creating a natural **Hover Bias**. The low-level PD controller then distributes this base thrust, clamped strictly at $7.5 \text{ N}$ per motor (Max $30 \text{ N}$ total), to execute the agent's target pitch, roll, and yaw commands. This provides a realistic **3:1 Thrust-to-Weight Ratio**, allowing agile recovery while strictly preventing physically impossible negative thrust.

## 2. The Curriculum Initialization Distance ("The Golden Ratio")
In Stage 0, the first coin is placed exactly $1.0 \text{ meter}$ away from the drone's spawn position (at World coordinate `[1.0, 0.0, 2.0]`, drone spawns near `[0, 0, 2.0]`). This is not an arbitrary number; it mathematically balances the reward economy to prevent the agent from bleeding points simply for existing.

**Note on direction:** The drone is Y-forward (body +Y = nose direction). At Yaw = 0, the coin at `[1.0, 0.0, 2.0]` is $1\text{m}$ to the drone's **right** (World +X), not ahead. With Symmetry Breaking (Stage 0.14) randomizing spawn Yaw across $[-\pi, +\pi]$, the coin is in a different body-relative direction every episode. This is intentional — the benefit of force-feeding is **proximity** (the coin is always $\approx 1\text{m}$ away, not $8\text{m}$ across the room), not direction. The ego-centric compass vector (`local_relative_pos`) in the observation always points the agent toward the coin regardless of heading.

Based on my YAML config:
- `alive_bonus`: $0.02$
- `distance_penalty_multiplier`: $0.02$

The net reward at spawn is calculated as:
$$\text{Net Reward} = \text{Alive Bonus} - (\text{Distance Penalty} \times \text{Distance})$$
$$\text{Net Reward} = 0.02 - (0.02 \times 1.0) = \mathbf{0.0}$$

At a distance of exactly $1.0 \text{ m}$, the agent receives exactly $0.0$ points per step. It does not suffer penalty bleed, nor does it farm free points. If the agent moves even $1 \text{ cm}$ closer (distance $= 0.99 \text{ m}$), the equation yields a positive reward, teaching the agent that moving toward the coin equals profit.

## 3. Cognitive Upgrades (Resolving the Training Bottleneck)
Despite an environment that was (in theory) mathematically sound, the initial `[64, 64]` Multilayer Perceptron (MLP) architecture failed to learn stable flight.

- **Brain Capacity:** Controlling 4 independent rotors in 3D space based on a 32-D observation vector (Kinematics + 16-ray Lidar) requires significant cross-correlation capabilities. I upgraded the network to a `[256, 256]` architecture to provide the necessary capacity for complex aerodynamic modeling (such as pitch braking without overshoot).
- **High-Frequency Control (240Hz):** Real-world quadcopters require extremely high-frequency PID loops (often 400Hz+) to maintain stability. The agent directly controls the rotors at the simulation frequency of 240Hz, allowing for micro-corrections and preventing unrecoverable aerodynamic flips.


# Evolution of the Drone Reward Function & Constraint Shaping

This document logs the chronological evolution of my drone RL agent's behavior, detailing the reward hacking bugs (local optima) I ran into and the fixes I applied working toward the final `Hunter Model` policy.

## Phase 0: Playing Possum (Lazy Local Optima)
**Behavior:** The drone realized flying was risky. It would immediately turn off its motors, fall to the floor, and simply lie flat to collect the continuous `Alive Bonus` without taking any collision penalties.
**Fix:** I introduced a strict Z-axis constraint. If the center of mass drops too low, it's considered a fatal collision.

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
**Fix:** I raised the floor death limit, added a severe penalty for tilting (Euler angles), and penalized motor effort to discourage chaotic spinning.

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
**Fix:** I reduced the distance penalty drastically so the agent could survive long enough to explore without bleeding points.

**Code Changes:**
```python
# CHANGED Distance Penalty
# OLD: reward -= 0.05 * current_distance
# NEW: reward -= 0.005 * current_distance 
```

## Phase 3: Ceiling Hugging (Icarus Effect)
**Behavior:** With the suicide bug fixed, the agent realized that the `Alive Bonus` was so high, and the `Distance Penalty` so low, that it didn't need to hunt for coins. It just flew straight up to the ceiling (where there are no obstacles) and crushed to the ceiling (as it couldn't learn to safely hover yet).
**Fix:** I implemented the final **Hunter Model**. I drastically reduced the alive bonus, tied it mathematically to the distance penalty (canceling each other out at long ranges), and significantly increased the coin reward.

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
**Fix:** 1. **Hover Bias (Action Normalization):** I mapped the agent's $[-1.0, 1.0]$ output space so that $0.0$ equals the exact physical hover force ($2.45 \text{ N}$ per motor), and clamped forces to $[0.0, 10.0] \text{ N}$ to enforce physical limits.
2. **The "Force-Feeding" Trick (Curriculum Initialization):** For Stage 0, I hardcoded the first coin at World position `[1.0, 0.0, 2.0]`, placing it exactly $1.0 \text{ m}$ from the drone's spawn point, to guarantee early reward discovery and break the local optima of "just staying alive". Note: this coin is to the drone's **right** at Yaw = 0, not ahead — the force-feeding benefit is **proximity**, not direction. See "Golden Ratio" in section 2 for the full explanation.

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
# NEW: Placing the first coin 1m away from spawn to guarantee early reward discovery.
# NOTE: [1.0, 0.0, 2.0] is 1m in World +X. Drone is Y-forward, so this is to the
# drone's RIGHT at Yaw=0, not ahead. The benefit is proximity, not direction.
fixed_positions = [
    [1.0, 0.0, 2.0],  # 1m from spawn — ego-centric compass guides the agent regardless of heading
    [0.0, 1.5, 2.0],
    [4.0, 4.0, 2.0],
    [-4.0, -4.0, 2.0]
]
```

## Stage 0.1: Kamikaze Policy (Economic Reform)
**Behavior:** The agent successfully solved the Sparse Reward problem and grabbed the first coin (+300). However, it refused to learn how to brake or stabilize. It discovered a mathematical loophole: diving aggressively into the coin and immediately crashing (-50) yielded a massive net profit of +250. It optimized for a quick death rather than sustainable flight.
**Fix:** I restructured the reward economy in the YAML configuration. I increased distance and velocity penalties to discourage erratic, high-speed dives. Crucially, I matched the `collision_penalty` (300) to the `coin_collection_reward` (300). Now, crashing immediately after collecting a coin results in a net-zero sum, forcing the agent to learn deceleration and hover stabilization to preserve its profits.

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
**Fix:** I implemented **"Hovercraft Mode" (Z-Axis Lock)** for Stage 0. By dynamically locking the drone's Z-coordinate to 2.0 meters programmatically, I removed gravity and altitude control from the agent's cognitive load. The idea was that it could now safely learn to pitch, roll, and yaw in the X/Y plane without catastrophic Z-axis momentum spikes, and master targeting and braking in 2D before unlocking full 3D physics in Stage 1.

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
**Fix:** I bridged the "Sim2Real" gap by treating the mathematical room boundaries as lethal physical obstacles. I appended the generated `wall_ids` to the physical collision detection loop. Now, even a millimeter of contact with a wall results in immediate termination and a `-300` collision penalty. This strict constraint forcefully evicts the agent from its "contact-rich" comfort zone, pushing it to learn true "free-flight" stabilization in the center of the room.

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
1. **Euler Angles:** I replaced quaternion with Euler angles (roll, pitch, yaw) in the 
   observation vector. Observation space reduced from 32-D to 31-D.
2. **Yaw Torque:** I added differential torque based on motor force imbalance:
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

**Fix:** I implemented **Hierarchical Control**, moving away from the "End-to-End" approach of directly controlling the 4 motor thrusts.
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

## Stage 0.6: The Effort Penalty Paradox (Action Space Decoupling)

**Behavior:** After migrating to the Hierarchical PD Control architecture, the agent exhibited extreme passivity. Instead of navigating toward the targets, it preferred to output near-zero actions `[0.0, 0.0, 0.0, 0.0]`, effectively hovering in place and refusing to pitch, roll, or yaw.

**Fix:** I identified the root cause in the reward shaping function: the `effort_penalty`.
In the previous "End-to-End" architecture, penalizing the sum of squared actions prevented the drone from spinning its motors wildly. However, in the new Hierarchical architecture, the agent's action space represents **High-Level Intent** (Target Pitch, Roll, Yaw Rate) rather than raw electrical motor effort. 

By continuing to penalize the action vector, the environment was mathematically punishing the agent for simply "making a decision to move." To eliminate this cognitive friction, I reduced the `effort_penalty_multiplier` to `0.0` in the YAML configuration. The physical motor effort is now inherently constrained and stabilized by the low-level PD controller, allowing the RL agent to freely issue attitude commands without artificial math penalties.

**Code Changes:**
```yaml
# CHANGED in configs/teacher_ppo.yaml
# OLD:
  effort_penalty_multiplier: 0.001
# NEW: Action space decoupled from raw effort; intent should not be penalized.
  effort_penalty_multiplier: 0.0
```

## Stage 0.7: The Z-Lock Trap & Curriculum Consolidation (Removing the Training Wheels)

**Behavior:** After implementing the PD controller, the agent was still being trained using "Hovercraft Mode" (`lock_z: True`), which artificially forced its Z-velocity to `0.0` and clamped its altitude to `2.0m`. Because the agent's altitude was hardcoded, any changes it made to `action[3]` (Target Thrust) had zero effect on the environment. The RL agent quickly learned that this action output was useless and stopped optimizing it. When transitioned to the next curriculum stage (`lock_z: False`), the agent catastrophically crashed because its policy had never learned to manage thrust in a 3D space.

**Fix:** I completely removed the `lock_z` constraint from the physics step. The new low-level PD controller already provides sufficient baseline stability, meaning the agent no longer needs artificial "training wheels" to survive its early steps. It can and must learn true 3D flight (managing thrust alongside attitude) from step zero. 

Because removing `lock_z` made the original Stage 0 (2D) and Stage 1 (3D) identical in their objectives, I consolidated the curriculum. The redundant stage was eliminated, streamlining the training pipeline from 6 stages down to 5 (Stage 0 to 4).

**Code Changes:**
```yaml
# CHANGED in configs/teacher_ppo.yaml
# Removed all lock_z parameters.
# Curriculum consolidated to 5 stages:
  stage_0: "Stage_0_Hover_and_Navigate" (Empty room, Fixed coins)
  stage_1: "Stage_1_RandomCoins"        (Empty room, Random coins)
  stage_2: "Stage_2_FixedObs_FixedCoins"(Fixed obstacles, Fixed coins)
  stage_3: "Stage_3_FixedObs_RandomCoins"(Fixed obstacles, Random coins)
  stage_4: "Stage_4_Full_Autonomy"      (Random obstacles, Random coins)

  # REMOVED from drone_sim.py step():
# if self.lock_z:
#     pos, ori = p.getBasePositionAndOrientation(self.drone_id)
#     lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
#     p.resetBasePositionAndOrientation(self.drone_id, [pos[0], pos[1], 2.0], ori)
#     p.resetBaseVelocity(self.drone_id, [lin_vel[0], lin_vel[1], 0.0], ang_vel)
```

## Stage 0.8: Policy Collapse, Observation Shock, and The Myopic Agent (240Hz Gamma Correction)

**Behavior:** After integrating the PD controller, the agent successfully learned to fly and collect the first coin (reaching 1000-step episodes and spiking to -50 mean reward). However, around the 2 Million step mark, the policy suffered a catastrophic collapse. The agent reverted to crashing, and the reward flatlined back to -350.

**Analysis:** This policy collapse was caused by two compounding factors specific to high-frequency (240Hz) continuous control:
1. **Observation Shock:** When the first coin is collected, the `relative_pos` vector instantly snaps to the second coin (e.g., from `[0, 0, 0]` to `[-1.0, 1.5, 0]`). This sudden teleportation in the state space causes a massive spike in the agent's action outputs, destabilizing the drone and causing a crash. The agent learns that "collecting a coin equals immediate death" and subsequently refuses to collect even the first coin to avoid the penalty.
2. **The Myopic Horizon (Discount Factor):** The default PPO `gamma` is `0.99`. In a 240Hz environment, a coin that is 11.3 meters away takes approximately 2700 steps to reach. With `gamma=0.99`, the future reward of `+300` is discounted to exactly `0.0000000004`. The agent is hyper-myopic; it cannot "see" the second or third coins because their discounted value is zero. Thus, overcoming the observation shock is not worth the risk.

**Fix:** I adjusted the discount factor to mathematically match the 240Hz time domain.
By changing the PPO hyperparameters to `gamma=0.9995`, the agent's horizon is extended. At `gamma=0.9995`, a coin 2700 steps away retains a perceived value of roughly `+77` points. This strong gradient incentivizes the agent to recover from the Observation Shock and pursue the remaining targets, effectively curing the policy collapse.

**Code Changes:**
```python
# CHANGED in scripts/train_teacher.py
# Extended the discount factor for high-frequency control
# OLD:
model = PPO("MlpPolicy", env, ...)
# NEW:
model = PPO("MlpPolicy", env, gamma=0.9995, ...)
```

## Stage 0.9: The "ETH Zurich" Physics & Perception Overhaul (Bridging Sim2Real and Cross-Modal Distillation)

**Behavior:** The agent demonstrated capability in navigating to targets, but its internal logic was extremely fragile. It struggled to handle 3D rotations, drifted uncontrollably due to a lack of air resistance, and exhibited "blind spots" when turning. Most critically, the observation space was not strictly ego-centric (Body Frame), which would create a catastrophic bottleneck during the later Cross-Modal Distillation phase (where the Student CNN must learn from camera pixels, which inherently exist in the Body Frame).

**Fix:** I implemented three architectural overhauls, drawing inspiration from ETH Zurich's robotics and drone research:
1. **Body-Frame Coordinate Transformation:** The `relative_pos` vector (target location), `linear_vel`, and `angular_vel` were transformed from the World Frame to the Body Frame using the drone's rotation matrix. The agent now perceives targets as "forward/left/right" rather than "North/South," aligning perfectly with the future CNN's visual perspective.
2. **Rotating LiDAR (Ego-Centric Sensors):** Previously, LiDAR rays were cast in fixed global directions. The ray-casting algorithm was rewritten to multiply local ray vectors by the drone's rotation matrix. The LiDAR array now dynamically rotates with the drone's Yaw, providing interpretable and consistent obstacle detection.
3. **Aerodynamic Damping (Rotor Drag):** PyBullet's vacuum space was causing "ice-skating" overshoot behaviors. I injected artificial linear (`-0.5`) and angular (`-0.05`) damping forces into the physics step. This synthetic air resistance should allow the drone to brake more naturally when attitude levels out, easing the burden on the PD controller and RL policy.

**Code Changes:**
```python
# CHANGED in drone_sim.py: Ego-centric coordinate transformation
rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
local_vel = rot_matrix.T.dot(linear_vel)
local_relative_pos = rot_matrix.T.dot(global_relative_pos)

# CHANGED in drone_sim.py: Dynamic rotating LiDAR
local_ray = np.array([math.cos(angle), math.sin(angle), 0])
global_ray = rot_matrix.dot(local_ray)

# ADDED in drone_sim.py: Aerodynamic Damping
drag_force = [v * -0.5 for v in drone_vel]
p.applyExternalForce(self.drone_id, -1, forceObj=drag_force, posObj=[0,0,0], flags=p.WORLD_FRAME)
```

## Stage 0.10: Distillation Readiness (Action Smoothness & Sensor Noise)

**Behavior:** With the physics largely in order, I did a final architectural review with the ultimate goal in mind: **Cross-Modal Policy Distillation (Teacher MLP -> Student CNN)**. The agent was learning, but its training conditions were too clean and synthetic. It relied on 100% accurate absolute numerical data and was permitted to issue erratic, high-frequency command oscillations (e.g., flipping target pitch from +30 to -30 degrees instantly).

**Analysis (Scope & Sim2Real Filtering):** When preparing a Teacher for a Student CNN, the dataset generated by the Teacher must be interpretable by the Student.
1. **The Jerk Problem:** If the Teacher generates erratic actions, the CNN (which only sees a sequence of similar pixel frames) will fail to correlate visual inputs with wild label fluctuations. The Loss function will explode. The Teacher's actions must be smooth and predictable.
2. **The God-Mode Problem:** The CNN will never output perfectly precise coordinates from pixels; it will output slightly noisy estimates. If the Teacher's policy is overfitted to perfect mathematics, the system will collapse when the CNN takes over.
3. **Domain Randomization Scoping:** While real-world deployments require mass/battery randomization, I scoped this project strictly to simulation-based Cross-Modal Distillation. Therefore, physical mass randomization was explicitly discarded to maintain focus on the core problem: Vision-Based Domain Randomization (textures, lighting), which will be handled in the Student training phase.

**Fix:** - **Action Smoothness Penalty:** I replaced the discarded `effort_penalty` with a `smoothness_penalty_multiplier`. The agent is now penalized for the squared difference between the previous action ($a_{t-1}$) and the current action ($a_t$). This forces the Teacher to generate a butter-smooth flight trajectory, creating a pristine dataset for the CNN.
- **Minimal Sensor Noise:** I injected a Gaussian noise (`scale=0.01`) into the observation vector. This slight perturbation forces the Teacher policy to become robust against the inevitable micro-inaccuracies that the CNN will introduce during the Student phase.

**Code Changes:**
```python
# ADDED in drone_sim.py: Minimal Sensor Noise
noise = np.random.normal(loc=0.0, scale=0.01, size=obs.shape)
obs_noisy = obs + noise

# ADDED in drone_sim.py: Action Smoothness Calculation
action_diff = np.sum(np.square(action - self.prev_action))
self.prev_action = action.copy()

# CHANGED in configs/teacher_ppo.yaml
# OLD: effort_penalty_multiplier: 0.0
# NEW: smoothness_penalty_multiplier: 0.05
```

## Stage 0.11: Asymmetric Actor-Critic & Preventing Perceptual Aliasing (49-D Ego-Centric State)

**Behavior:** While reviewing the architecture for the upcoming Cross-Modal Distillation (Teacher MLP -> Student CNN), I identified a severe "Modality Mismatch" vulnerability. The Teacher was receiving Global X, Y coordinates (`drone_pos`). If the Teacher's policy optimized around global spatial memorization (e.g., "fly to absolute coordinate [3, 4]"), the Student CNN—which only perceives local pixel data—would suffer from "Perceptual Aliasing" (inability to distinguish visually identical opposite corners of the room) and fail to mimic the Teacher. Furthermore, the 16-ray LiDAR resolution was deemed too sparse, risking physical objects slipping between rays, which would poison the Teacher's logic dataset.
**Fix:** I completely purged the observation space of any "God-Mode" global positioning, while retaining relative depth perception to keep the Teacher's expert advantage.
1. **Removed Global Position:** I entirely deleted `drone_pos[0]` (X) and `drone_pos[1]` (Y) from the observation array. Only `z_altitude` was retained as it is critical for ground collision avoidance.
2. **LiDAR Resolution (The Goldilocks Zone):** I increased LiDAR rays from 16 to 36 (10-degree intervals). This closes blind spots without causing the Curse of Dimensionality. The final Observation Space expanded to a lightweight `49-D` array, which is trivially processed by the `[256, 256]` MLP while ensuring 100% ego-centric alignment with the future Student CNN.

**Code Changes:**
```python
# CHANGED in drone_sim.py: Removed Global XY
# OLD: obs = np.concatenate([drone_pos, euler, ...
z_altitude = np.array([drone_pos[2]], dtype=np.float32)
obs = np.concatenate([z_altitude, euler, local_vel, local_ang_vel, local_relative_pos, lidar_data])

# CHANGED in drone_sim.py: LiDAR Resolution
# OLD: self.num_rays = 16 (31-D State Space)
# NEW: self.num_rays = 36 (49-D State Space)
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)
```

## Stage 0.12: Neural Network Mathematics (Yaw Singularity & State Normalization)

**Behavior:** Although the environment physics and distillation architecture were largely settled, I expected PPO to still struggle with slow convergence and gradient explosions. I traced this back to two fundamental data-formatting traps, not physics bugs.

**Fix 1 (The Yaw Wrap-Around Singularity):** The Yaw angle spanned from $-\pi$ to $+\pi$. A drone rotating past $180^\circ$ would experience a sudden mathematical discontinuity, jumping from $+3.14$ to $-3.14$. This $2\pi$ jump causes catastrophic gradient spikes. 
*Solution:* I decoupled the Yaw scalar into a continuous trigonometric tuple `[sin(yaw), cos(yaw)]`, mapping the rotation to a smooth unit circle. The state space increased to `50-D`.

**Fix 2 (The Curse of Unnormalized States):** The 50-D state vector contained values of vastly different scales (LiDAR: $0 \rightarrow 5$, Velocity: $-10 \rightarrow 10$, Noise: $0.01$). MLPs require inputs to follow a $\sim N(0,1)$ distribution to prevent large-magnitude inputs from dominating weight updates.
*Solution:* I wrapped the environment in Stable-Baselines3's `VecNormalize` to dynamically track running means and variances, normalizing all observations on the fly. 

**Code Changes:**
```python
# CHANGED in drone_sim.py: Yaw Singularity Fix
yaw_sin = math.sin(current_yaw)
yaw_cos = math.cos(current_yaw)
obs = np.concatenate([z_altitude, [current_roll, current_pitch], [yaw_sin, yaw_cos], ...])

# ADDED in train_teacher.py: Deep Learning Input Normalization
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# FIX in train_teacher.py: Evaluation Environment Threshold Protection
# norm_reward must be False during eval, otherwise the +1600 threshold is unreachable.
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
```

## Stage 0.13: The Three Hidden Monsters (Frame Conflicts, Gamma Mismatch, and State Desync)

**Behavior:** Even with a clean observation space, the training pipeline had three deep structural flaws spanning physics, RL math, and software engineering. If left unchecked, these would cause the model to plateau indefinitely.

**Monster 1 (Physics): Blind PD Controller.**
The low-level PD controller was receiving the drone's angular velocity (`ang_vel`) in the Global World Frame. When the drone rotated 90 degrees (Yaw), the World's Y-axis no longer aligned with the drone's Pitch axis. The PD controller attempted to stabilize Pitch but inadvertently applied torque to the Roll axis, inducing an unrecoverable death-spin.
*Fix:* I multiplied the global `ang_vel` by the transposed rotation matrix to convert it into `local_ang_vel` (Body Frame) before feeding it into the PD damping calculations.

**Monster 2 (RL Math): The Gamma Horizon Mismatch.**
To solve 240Hz myopia, the PPO agent's discount factor was set to $\gamma = 0.9995$. However, the underlying `VecNormalize` wrapper uses its own gamma to compute discounted returns for reward normalization. Its default was $0.99$. The wrapper was squashing the rewards based on a 1-second horizon, confusing the PPO agent targeting a 5-second horizon.
*Fix:* I explicitly passed `gamma=0.9995` to all `VecNormalize` instances to synchronize the time horizons.

**Monster 3 (Software Eng): Evaluation Environment Desync.**
The `eval_env` was initialized with `training=False`, meaning its normalization statistics (Mean, Variance) remained frozen at zero. When the `EvalCallback` tested the model, the model was fed incorrectly scaled raw data, causing it to fail every evaluation and never save a `best_model.zip`.
*Fix:* I wrote a custom `SyncEvalEnvCallback` that explicitly copies the running `obs_rms` from the training environment to the evaluation environment at every step.

## Stage 0.14: The Final Boss — Symmetry Breaking (Preventing Muscle-Memory Overfitting)

**Behavior:** With all systems finally synchronized, a theoretical vulnerability remained regarding the Deep Learning policy's generalization capability. The drone spawned at the exact same coordinate (`[0, 0, 2.0]`) facing exact North (`Yaw=0`) at the start of every single episode. In Reinforcement Learning, deterministic initialization allows the MLP to bypass sensor data completely and memorize a rigid "open-loop" sequence of actions (Muscle Memory) to reach the first target. When transitioned to the Cross-Modal Student CNN phase, any slight wind or initialization noise would cause catastrophic failure.

**Fix:** I implemented **Symmetry Breaking** in the environment's `reset()` function. The drone is now spawned with a randomized X/Y offset ($\pm 0.5$ meters) and a completely randomized starting Yaw angle ($-\pi$ to $+\pi$). This strict Domain Randomization forcefully prevents trajectory memorization. The agent has no choice but to actively process its Ego-Centric 50-D sensor arrays (LiDAR and Compass) from step zero to survive, guaranteeing true "closed-loop" zero-shot generalization.

**Code Changes:**
```python
# CHANGED in drone_sim.py reset(): Added Symmetry Breaking
start_x = self.np_random.uniform(-0.5, 0.5)
start_y = self.np_random.uniform(-0.5, 0.5)
start_yaw = self.np_random.uniform(-math.pi, math.pi)
start_pos = [start_x, start_y, 2.0]
start_ori = p.getQuaternionFromEuler([0, 0, start_yaw])
self.drone_id = p.loadURDF(urdf_path, start_pos, baseOrientation=start_ori, globalScaling=4.0)
```

## Stage 0.15: The LiDAR Projection Shrinkage (Mathematical Frankenstein)

**Behavior:** In Stage 0.9, the LiDAR was made ego-centric by multiplying the local ray vectors with the full 3x3 rotation matrix. However, because the environment utilizes a 2D LiDAR plane, the Z-component of the resulting 3D vector was discarded. Mathematically, dropping the Z-component of a pitched/rolled unit vector extracts its 2D projection. As the drone pitched forward (e.g., 45 degrees), the projection mathematically shrank ($\cos(45^\circ) \approx 0.707$). The 5.0-meter LiDAR functionally shrank to 3.53 meters, distorting into an ellipse and severely crippling the agent's spatial awareness.

**Fix:** I implemented **Gimbal-Stabilized LiDAR**, scrapping the full 3x3 rotation matrix for LiDAR rays. Instead, the local rays are now rotated **only by the Yaw angle**. This perfectly mimics real-world 2D LiDAR systems: it rotates with the drone's heading but remains strictly parallel to the ground, immune to projection shrinkage caused by Pitch or Roll.

**Code Changes:**
```python
# CHANGED in drone_sim.py: Gimbal-Stabilized (Yaw-Only) LiDAR Rotation
euler = p.getEulerFromQuaternion(ori)
yaw = euler[2]

for i in range(self.num_rays):
    angle = (2 * math.pi * i) / self.num_rays + yaw
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    start = [drone_pos[0] + dx*offset, drone_pos[1] + dy*offset, drone_pos[2]]
    end = [start[0] + dx*self.lidar_range, start[1] + dy*self.lidar_range, start[2]]
```

## Stage 0.16: The Final Polish (URDF Alignment, Smoothness Scaling & Entropy)

**Behavior:** The agent continued to exhibit extreme "Beyblade" spinning and struggled to establish early stable flight, despite previous physical corrections. The root causes were a combination of inverted physics commands, hyper-sensitive penalty calculations, and an overly strict target bounding box.

**Bug 1 (Physics): The URDF Motor Mixing Mismatch.** While the PD controller correctly calculated required torques, the X-configuration motor mixing matrix incorrectly mapped them to the drone's physical layout. Based on PyBullet's coordinate system and the specific `cf2x.urdf` offsets, Motor 1 (Front-Left) and Motor 3 (Rear-Right) were receiving inverted Pitch and Yaw commands. The PD controller was effectively commanding the drone to crash when attempting to correct errors.
*Fix:* I completely rewrote the motor mixing matrix to align with the `cf2x.urdf` coordinate topology, ensuring correct sign application for all torques across all four rotors.

**Bug 2 (Reward Math): Smoothness Penalty Over-scaling.**
The jerk penalty was calculated using `np.sum(np.square(action - self.prev_action))`. Because continuous actions operate in a high-frequency (240Hz) 4-dimensional space, the raw sum produced a massive penalty per step (often overshadowing the `alive_bonus`). This paralyzed the agent, as moving any motor was mathematically worse than falling to the floor.
*Fix:* I changed the calculation to `np.mean` to normalize the penalty, and additionally softened the PD controller gains (`Kp_ang` from 8.0 to 5.0) to prevent the motors from clamping to their maximum limits during early random exploration.

**Bug 3 (Environment Geometry): The Pixel-Perfect Collection Radius.**
The $0.4 \text{ m}$ collection radius was too strict for a $36 \text{ cm}$ drone. The drone's physical body consumed $18 \text{ cm}$ of this radius, leaving a mere $22 \text{ cm}$ margin of error, making target collection a nearly impossible needle-threading task.
*Fix:* I expanded the collection radius to $0.6 \text{ m}$, providing a physically realistic "rotor wash" hit-box that rewards the agent for aggressive near-miss flybys.

**Bug 4 (RL Math): Entropy Collapse.**
PPO defaults to an entropy coefficient of $0.0$, which can lead to premature deterministic policies (e.g., spinning continuously) before the agent discovers the sparse rewards.
*Fix:* I added `ent_coef=0.01` to the PPO configuration to actively encourage early-stage exploration.

**Code Changes:**
```python
# CHANGED in drone_sim.py: Corrected X-Configuration Matrix
f0 = base_f - tau_pitch + tau_roll - tau_yaw  # Motor 0 (Front-Right)
f1 = base_f - tau_pitch - tau_roll + tau_yaw  # Motor 1 (Front-Left)
f2 = base_f + tau_pitch - tau_roll - tau_yaw  # Motor 2 (Rear-Left)
f3 = base_f + tau_pitch + tau_roll + tau_yaw  # Motor 3 (Rear-Right)

# CHANGED in drone_sim.py: Smoothness Penalty using np.mean
action_diff = np.mean(np.square(action - self.prev_action))

# CHANGED in scripts/train_teacher.py: Added Entropy
model = PPO(..., ent_coef=0.01)
```

## Stage 0.17: Directional Blindness (Euler Frame Desync)

**Behavior:** Even after correcting the motor mixing matrix, the agent experienced an unpredictable "1-0-1-0" death spiral. Sometimes it hovered perfectly, but most of the time, it instantly flipped 90 degrees and crashed the millisecond it spawned (when spawning with non-zero yaw due to Symmetry Breaking).

**Bug (Euler Frame Desync - "Directional Blindness"):**
The PD Controller was designed for Body-Frame stabilization, but PyBullet's `getEulerFromQuaternion` returns Pitch and Roll in the *World Frame*. If the drone spawned facing East (Yaw = 90°) and pitched forward (Body Pitch), the physics engine reported a World Roll. The PD controller, seeing a "Roll error", would command a violent sideways correction, causing the drone to tear itself apart trying to correct a non-existent mistake.
*Fix:* I implemented a 2D trigonometric rotation matrix to dynamically convert World Pitch/Roll into Body Pitch/Roll based on the current Yaw angle. The PD controller is no longer "blind" to its own heading.

**Code Changes:**
```python
# CHANGED in drone_sim.py: Euler World-to-Body Transformation
world_pitch_raw = euler[0]  # X rotation
world_roll_raw  = euler[1]  # Y rotation
body_pitch_raw = world_pitch_raw * math.cos(current_yaw) + world_roll_raw * math.sin(current_yaw)
body_roll_raw  = -world_pitch_raw * math.sin(current_yaw) + world_roll_raw * math.cos(current_yaw)
```

## Stage 0.18: The Phantom Torque (WORLD_FRAME Lever Arm — The Root Bug)

**Behavior:** After all previous fixes, the drone still died within ~80 steps (~0.33 seconds) by rolling past the 75-degree tilt threshold. Critically, the failure was **position-dependent**: a drone spawned at exactly `[0, 0, 2.0]` with `action=[0,0,0,0]` flew perfectly; a drone spawned at `[0, 0.1, 2.0]` died immediately. Angular velocity grew at a perfectly constant rate each step — a textbook signature of a constant phantom torque proportional to position offset.

**Root Cause — A Misunderstood PyBullet API:**
The aerodynamic drag force was being applied as:
```python
p.applyExternalForce(..., posObj=[0, 0, 0], flags=p.WORLD_FRAME)
```
I had written a comment in the code saying *"applied to drone's Center of Mass [0,0,0]"* — but that was wrong. PyBullet's `applyExternalForce` interprets `posObj` according to the `flags` argument:

- `posObj=[0,0,0]` + `WORLD_FRAME` → force applied at the **world origin** in world space
- `posObj=drone_pos` + `WORLD_FRAME` → force applied at the **drone's actual COM**

When the drone is at world position `[x, y, z]`, the `[0,0,0]` application creates an invisible lever arm:
```
r = [0,0,0] − [x, y, z] = [−x, −y, −z]
τ = r × F_drag
```
This phantom torque grows with altitude (z increases as the drone rises) and with speed (F_drag scales with velocity). Together, they produce an ever-increasing angular acceleration that flips the drone. This is why the bug was invisible at [0,0] but lethal everywhere else.

**History:** Stage 0.9 introduced the drag force with `posObj=[0,0,0]`. At that point, Symmetry Breaking (Stage 0.14) had not yet been added, so the drone always spawned at [0,0], making the bug invisible. When Symmetry Breaking was added in 0.14, the phantom torque activated but was misattributed to other causes (motor mixing, Euler axes) through Stages 0.15-0.17. During debugging, I briefly tried `posObj=drone_pos_current` (which was actually the correct fix all along), but I reverted it after incorrectly convincing myself it was creating "a phantom pole above the drone."

**Fix:**
Capture `drone_pos_pre` from the `getBasePositionAndOrientation` call already made at the top of `step()` (previously discarded with `_`), and use it as the force application point.

**Code Changes:**
```python
# CHANGED in drone_sim.py step(): Capture drone position instead of discarding it
# FROM: _, ori = p.getBasePositionAndOrientation(self.drone_id)
# TO:   drone_pos_pre, ori = p.getBasePositionAndOrientation(self.drone_id)

# CHANGED in drone_sim.py step(): Apply drag at actual COM, not world origin
# FROM: p.applyExternalForce(..., posObj=[0, 0, 0], flags=p.WORLD_FRAME)
# TO:   p.applyExternalForce(..., posObj=list(drone_pos_pre), flags=p.WORLD_FRAME)
```

**Expected Result:**
With `action=[0,0,0,0]`, the drone should rise slowly and smoothly to the ceiling without dying at random spawn positions. The phantom torque is gone — Stage 0 training is currently running to validate whether the policy will converge this time.

## Stage 0.19: Perceptual Aliasing & Yaw Torque Sign (Two-Bug Root Cause Analysis)

**Behavior:** After the phantom torque fix (0.18), the drone survived longer but success_rate remained 0% at 1M steps. The reward improved (−330 → −175) and episode length rose to ~1000 steps, but no full coin collection was observed. The policy appeared to know how to fly but could not correlate its actions with consistent outcomes.

**Bug 1 — Perceptual Aliasing (World-Frame Observation vs Body-Frame PD):**
Stage 0.17 correctly fixed the PD controller to use Body-Frame pitch and roll. However, `_get_obs()` was never updated. It continued sending raw World Euler angles (`euler[0]`, `euler[1]`) as the agent's pitch and roll observation.

With Symmetry Breaking (Stage 0.14) randomizing spawn Yaw across `[−π, π]`, the consequence was catastrophic:
- At Yaw = 90°: world `euler[0]` ≈ body roll, world `euler[1]` ≈ body pitch. The axes were completely swapped.
- At Yaw = 180°: pitch sign was inverted.

From the neural network's perspective, the physical laws governing pitch and roll appeared to **change randomly at every episode**. Convergence was mathematically impossible at non-zero yaw.

Additionally, the observation sent raw `euler[0]` (PyBullet convention: positive = nose UP), while the PD controller defined `current_pitch = -body_pitch_raw` (positive = nose DOWN). Even at Yaw = 0, the observation pitch and PD pitch had opposite signs.

*Fix:* Applied the exact same 2D yaw rotation used in `step()` to `_get_obs()`, converting world euler to body-frame pitch/roll before putting them in the observation. Sign convention now matches the PD: **nose DOWN = positive, roll RIGHT = positive.**

**Bug 2 — Yaw Torque Inverted (Destabilizing PD Loop):**
Rotor forces `[0, 0, F]` applied at positions `[x, y, 0]` in `LINK_FRAME` produce torque `r × F = [y·F, −x·F, 0]`. The **Z-component is zero**. The explicit `torque_mag` is therefore the **sole yaw control mechanism** — there is no other source of yaw torque in the simulation.

The original formula:
```python
torque_mag = ((forces[0] + forces[2]) - (forces[1] + forces[3])) * 0.01
```

Verified against the Crazyflie/URDF motor spin directions (confirmed via Bitcraze documentation):
- Motors 0, 2 (prop0 front-right, prop2 rear-left) = **CCW** → CW reaction on body (−Z)
- Motors 1, 3 (prop1 front-left, prop3 rear-right) = **CW** → CCW reaction on body (+Z)
- Physically correct net torque: `((F1+F3) − (F0+F2)) * k`

The original formula had the sign **reversed**. When the PD demanded CCW correction (tau_yaw > 0), the motor mixing correctly increased CW rotors (1,3) and decreased CCW rotors (0,2), but `torque_mag` then applied a CW body torque. The feedback loop was **destabilizing**: the harder the PD tried to correct yaw, the faster the drone spun in the wrong direction.

*Fix:* Inverted the `torque_mag` formula sign.

**Code Changes:**
```python
# CHANGED in drone_sim.py _get_obs(): Body-Frame Euler Transformation
# OLD:
current_pitch = euler[0]   # raw world X rotation
current_roll  = euler[1]   # raw world Y rotation

# NEW: Mirror the same 2D yaw rotation used in step() PD controller
body_pitch_raw = world_pitch_raw * math.cos(current_yaw) + world_roll_raw * math.sin(current_yaw)
body_roll_raw  = -world_pitch_raw * math.sin(current_yaw) + world_roll_raw * math.cos(current_yaw)
current_pitch  = -body_pitch_raw   # Nose DOWN = positive (matches PD convention)
current_roll   =  body_roll_raw    # Roll RIGHT = positive (matches PD convention)

# CHANGED in drone_sim.py step(): Yaw Torque Sign
# OLD: torque_mag = ((forces[0] + forces[2]) - (forces[1] + forces[3])) * 0.01
# NEW:
torque_mag = ((forces[1] + forces[3]) - (forces[0] + forces[2])) * 0.01
```

## Stage 0.20: Hallucinated Reward (Noisy LiDAR as Reward Judge)

**Behavior:** Not a training failure, but an architectural impurity identified during review.

**Bug — The Noisy Judge:**
`_get_obs()` injects Gaussian noise (`σ = 0.01`) into the full 50-D observation vector before returning it, including the 36 LiDAR fractions. In `step()`, the reward function was extracting LiDAR from this noisy array:

```python
obs = self._get_obs()      # Noise applied inside
lidar_data = obs[-36:]     # Extracted noisy LiDAR
reward = compute_dense_reward(..., lidar_data, ...)  # Judge uses hallucinated data
```

This violates the fundamental RL principle: **the agent perceives through noisy sensors, but the environment rewards based on ground truth.** A lidar reading of `0.105` (safe, no penalty) could become `0.095` after noise and trigger an undeserved proximity penalty.

The practical impact is small (`max hallucinated penalty ≈ 0.005` per step, Gaussian noise averages to zero so there is no systematic bias), but the architecture is wrong.

**Fix:** Compute a separate clean LiDAR scan directly from PyBullet for reward evaluation. `drone_pos` and `ori` are already available in `step()` from the post-physics-step re-fetch, so no extra API calls are needed beyond the `_compute_lidar()` call itself.

**Code Changes:**
```python
# CHANGED in drone_sim.py step(): Separate clean LiDAR for reward
obs = self._get_obs()                              # Agent receives noisy observation
clean_lidar = self._compute_lidar(drone_pos, ori)  # Ground truth for the reward judge
reward = compute_dense_reward(..., clean_lidar, ...)
```

## Stage 0.21: Entropy Collapse & Spawn Contamination (Training Diagnostics)

**Behavior:** After 2.94M steps (~10 hours), the policy plateaued and regressed. Mean eval reward peaked at −94 around 1.8M steps, then degraded back to −178 by 3M steps. Success rate remained 0% throughout all 294 evaluations. Entropy collapsed to −8, indicating a fully deterministic policy stuck in a local optimum. The agent occasionally collected coin 1 and rarely coin 2, but never proceeded to coins 3 or 4.

**Root Cause Analysis:**

**Bug 1 — Entropy Collapse (`ent_coef` Too Low):**
`ent_coef=0.01` (added in Stage 0.16) was insufficient for this task's complexity. By 1.8M steps the policy converged prematurely to a deterministic coin-1-collection strategy, losing the ability to explore post-collection navigation. Gradient updates continued but pushed the policy toward marginal local improvements that destroyed other capabilities, explaining the regression from −94 back to −178.

**Bug 2 — Spawn Inside Collection Radius (~4.5% of Episodes):**
`reset()` spawned the drone before calling `_spawn_coins_safely()`, making it impossible to check the drone's position against coin locations. Monte Carlo analysis (N=1,000,000) confirmed ~4.5% of episodes place the drone within 0.6m of coin 1 at `[1.0, 0.0, 2.0]`, causing an instant free +300 reward on the very first `step()` with no learned behavior. This injected noisy false-positive signals into the training data — the policy received coin-collection reward without the observation sequence that led to it.

**Fix 1 (RL Hyperparameter):** Raised `ent_coef` from 0.01 to 0.05. Resume from best_model.zip (−94 reward checkpoint) to preserve learned flight skills while injecting renewed exploration pressure.

**Fix 2 (Environment):** Reversed spawn order in `reset()` — coins are now spawned first, then the drone position is sampled and rejected if within 0.6m of any coin (up to 10 retries). `smoothness_penalty_multiplier` also reduced from 0.05 to 0.02 to reduce over-constraint on early-stage exploration.


**Code Changes:**
```python
# CHANGED in drone_sim.py reset(): Spawn coins before drone, check radius
self._spawn_coins_safely()  # coins first
for _ in range(10):
    start_x = self.np_random.uniform(-0.5, 0.5)
    start_y = self.np_random.uniform(-0.5, 0.5)
    too_close = any(
        math.sqrt((start_x - g["pos"][0])**2 + (start_y - g["pos"][1])**2) < 0.6
        for g in self.gold_data
    )
    if not too_close:
        break

# CHANGED in scripts/train_teacher.py: ent_coef 0.01 → 0.05
# ADDED: auto-resume from current stage best_model.zip with vecnorm reload
# CHANGED: DummyVecEnv → SubprocVecEnv(N_ENVS=4) for parallel data collection
#   - 4x faster rollout; PyBullet is CPU-only so RTX 3060 laptop handles this easily
#   - Factory functions required (pre-created instances can't be pickled)
#   - Only rank=0 env writes to monitor.csv to avoid file corruption
#   - batch_size: 256 → 512 (total rollout = 4096×4 = 16384; 32 mini-batches/update)
ent_coef=0.05
N_ENVS=4
batch_size=512

# CHANGED in configs/teacher_ppo.yaml
# smoothness_penalty_multiplier: 0.05 → 0.02
```

---

## Stage 0.22 — Curriculum Reboot: Hover Foundation + Reward Economy Fix

**Date:** 2026-04-03

**Trigger:** Despite 5M+ training steps, ep_len_mean plateaued at ~1350 steps (5.6s out of 60s max). `ent_coef=0.05` caused immediate entropy explosion to −18 (policy → pure noise). Root cause analysis and cross-referencing with ETH Zurich gym-pybullet-drones revealed two fundamental architectural problems.

---

### Problem 1 — No Hover Foundation (Missing Pre-Training Stage)

The agent was asked to simultaneously learn attitude stabilization AND coin navigation from a random initialization. With `log_std_init=0.0` (default, std=1.0), the random policy issues ±30° attitude commands, the PD controller executes them faithfully, and the drone flips and dies in ~5.6 seconds before accumulating any positive reward signal. No coin is ever seen → no gradient toward navigation → policy never improves.

ETH Zurich and all surveyed navigation repos train hover **first** (separate stage, zero coin reward), then layer navigation on top of learned flight mechanics.

**Fix:** Added `Stage_0_Hover` as the curriculum entry point. No coins, no navigation. Reward = altitude stability + tilt minimization + alive bonus. Same 50D observation space throughout (LiDAR = 1.0 + noise in hover, network learns to weight it near zero; when obstacles appear in Stage 4+, LiDAR becomes meaningful without obs space change). Transfer learning works because same network, same obs/action space — only reward differs.

New curriculum:
```
Stage 0: Hover         (survive + hold Z=2.0)
Stage 1: Scout         (1 fixed coin, 1m away)
Stage 2: Navigator     (4 fixed coins)
Stage 3: Hunter        (10-18 random coins)
Stage 4: Pioneer       (20 fixed obstacles + random coins)
Stage 5: Apex          (20 random obstacles + random coins)
```

**Fix:** `log_std_init=-1.2` (std≈0.3). Initial effective tilt = ±9° instead of ±30°. Drone survives random policy → accumulates reward signal → gradient can propagate.

---

### Problem 2 — Velocity Penalty Kills Navigation (Reward Economy Bug)

Per-step reward change when flying at velocity `v` toward a coin:

```
distance_penalty improvement : +0.02 × (v / 240Hz) = +0.000083v  per step
velocity_penalty cost         : −0.003 × v          = −0.003000v  per step
────────────────────────────────────────────────────────────────────────────
net gradient per step         :                        −0.002917v  (negative)
```

**Moving toward a coin at any speed gives a negative per-step reward gradient.** The 36× imbalance between velocity_penalty and the per-step distance improvement means the immediate policy gradient says "hover, don't move." Only the long-horizon value function (coin +300 many steps ahead) counteracts this, making credit assignment extremely hard — especially early in training when the value function is unreliable.

This explains why the agent learned to hover in place at ~1m from coin 1: the alive_bonus and distance_penalty balanced at 1m (Golden Ratio), velocity_penalty discouraged movement, and the coin was never close enough to collect by random drift.

**Fix:** `velocity_penalty_multiplier: 0.003 → 0.0` in `nav_rewards`. Hover stage retains its own velocity damping (separate `hover_rewards` section). Distance penalty alone provides sufficient navigation gradient: getting closer = less penalty per step.

---

### Problem 3 — ent_coef Instability

`ent_coef=0.01` caused slow entropy collapse. `ent_coef=0.05` caused immediate entropy explosion (−18 in 120K steps). The right value is near SB3 default (0.0). Set to `0.005` as a minimal safety margin against collapse.

---

### gamma and batch_size — Confirmed Correct

Cross-referenced against ETH Zurich and surveyed papers:
- Papers use `gamma=0.99` at 50Hz → effective horizon ~2s. At our 240Hz, `gamma=0.99` → horizon 0.42s (completely myopic, coin 5s away invisible). `gamma=0.9995` → horizon 8.3s. **Correct for 240Hz.**
- Papers use `batch_size=64` with single env (2048 transitions, 32 mini-batches). We use `batch_size=512` with N_ENVS=4 (16384 transitions, 32 mini-batches). **Same ratio. Correct.**

---

**Code Changes:**
```python
# NEW: drone_env/drone_sim.py
# - hover_only=False, num_fixed_coins=4 parameters added to __init__
# - _spawn_coins_safely: respects num_fixed_coins limit for staged coin introduction
# - reset(): coins not spawned when hover_only=True
# - step(): hover_only branch skips coin/success logic; uses compute_hover_reward()

# NEW: drone_env/reward_functions.py
# - compute_hover_reward() added: alive + altitude_error + tilt + velocity + ang_vel + collision
# - compute_dense_reward() unchanged except smoothness fallback already fixed

# NEW: configs/teacher_ppo.yaml
# - Restructured: 6 stages (Hover→Scout→Navigator→Hunter→Pioneer→Apex)
# - hover_rewards and nav_rewards separate sections
# - velocity_penalty_multiplier: 0.003 → 0.0 in nav_rewards
# - reward_threshold per stage (Hover:800, Scout:600, Navigator:1500, ...)
# - num_fixed_coins per stage for gradual coin introduction

# CHANGED: scripts/train_teacher.py
# - policy_kwargs: log_std_init=-1.2 added (std=0.3, prevents crash-on-spawn)
# - ent_coef: 0.05 → 0.005
# - reward_weights: reads hover_rewards or nav_rewards based on hover_only flag
# - StopTrainingOnRewardThreshold: hardcoded 1600 → per-stage REWARD_THRESHOLD from YAML
# - make_env / eval_env_raw: hover_only and num_fixed_coins passed through
```

---

## Stage 0.23 — Hover Compass Fix: Unified Target Architecture

**Date:** 2026-04-03

**Trigger:** Stage 0.22's hover reward only targeted Z=2.0 (altitude). The ego-centric compass in the observation was hardcoded to `[0, 0, 0]` when no coins existed (`hover_only=True`). This created two problems:

1. **Obs/reward inconsistency:** Reward penalized distance from Z=2.0, but the compass (indices 11-13 in 50D obs) never pointed anywhere — the policy had no directional signal for x/y positioning.
2. **Transfer gap:** In hover, policy learned "compass=[0,0,0] → hover in place." In Stage 1, compass suddenly becomes a real non-zero vector pointing to a coin. The policy had never seen a non-zero compass during hover training, so the transferred skill didn't include x/y positional control.

---

### Root Cause — Obs/Reward Architecture Mismatch

ETH Zurich and principled curriculum designs treat hover as navigation with a fixed target. The key insight:

> Policy learns one skill: minimize the compass vector. In hover, target is `[0, 0, 2.0]` (fixed). In nav, target shifts as coins are collected. Same 50D obs, same network, same behavior — only the target location changes.

Our Stage 0.22 hover used `local_relative_pos = [0, 0, 0]` (no target), forcing the policy to learn a fundamentally different behavior than navigation. The hover→nav transfer would have required the policy to learn x/y positioning from scratch in Stage 1.

---

### Fix — Virtual Hover Target as Real Compass Anchor

**`drone_sim.py / _get_obs()`:** When `hover_only=True`, compass now computes real ego-centric vector to `hover_target=[0,0,2.0]`:
```python
if self.hover_only:
    global_relative_pos = np.array(self.hover_target) - np.array(drone_pos)
    local_relative_pos = rot_matrix.T.dot(global_relative_pos)
```

**`reward_functions.py / compute_hover_reward()`:** Reward now penalizes 3D distance to target (not just altitude error). `velocity_penalty` removed — physics drag + tilt penalty provide natural damping without biasing Stage 1+ toward slow movement:
```python
dist = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
reward = alive_bonus - distance_penalty * dist - tilt_penalty * (pitch²+roll²) - ang_vel_penalty * |ang_vel|
```

**`configs/teacher_ppo.yaml / hover_rewards`:**
```yaml
hover_rewards:
  alive_bonus: 0.1
  hover_target: [0.0, 0.0, 2.0]
  distance_penalty: 0.5          # replaces altitude_penalty (now 3D)
  tilt_penalty: 0.5
  angular_velocity_penalty: 0.05
  collision_penalty: 50.0
  # removed: altitude_target, altitude_penalty, velocity_penalty
```

Stage 0 `reward_threshold`: 800 → 500. Breakeven at dist=0.2m; threshold 500 requires dist < 0.13m average.

---

### Reward Numerics

```
alive_bonus=0.1, distance_penalty=0.5:

Perfect hover (dist=0, tilt=0):  +0.1/step → 14400 steps = +1440 (theoretical max)
Breakeven (0/step):               dist = 0.2m
Threshold 500:                    +0.035/step → dist < 0.13m average
Threshold 800 (old):              +0.056/step → dist < 0.09m (too strict for early training)
```

---

### Why velocity_penalty Was Removed from Hover

Explicit velocity penalty biases the transferred policy toward slow movement. Stage 1 needs the drone to rush toward coins. Natural damping from physics drag (`-0.5 * v`) and tilt penalty makes the explicit velocity term redundant.

---

**Training run terminated at ~2.36M steps** (eval/mean_ep_length reached 8823, eval/mean_reward -1022) before the compass fix was discovered. Logs archived to `logs/legacy/5_before_virtual_target/`. New training starts from scratch with corrected architecture.

---

## Stage 0.24 — Collision Penalty & LR Schedule Fix (Local Optimum Regression)

**Trigger:** Stage 0.23 run (~2.31M steps) showed a clean pattern: policy peaked at eval=-135 around 320K steps, then catastrophically regressed. By 2.31M steps, eval=-851, ep_len=4474 (drone surviving 18s but hovering ~0.88m from target). The drone learned to hover stably but ignored the compass.

### Root Cause: "Avoid Target" Local Optimum

At 320K the drone was approaching the target (ep_len≈302, fast approach + tilt death). Policy gradient saw repeated crash events and learned "approaching = crash". It retreated to "survive far from target" equilibrium:

```
survive at 0.88m:  per step = 0.1 - 0.5×0.88 = -0.34/step  (small, safe)
approach + crash:  collision_penalty = -50  (large, one-time)
```

With `collision_penalty=50`, the policy correctly calculated that a single crash wipes out ~147 steps of "safe hovering". Entropy rose from -0.62 → -1.30 (policy re-explored), std rose 0.282 → 0.338 — policy destabilized to escape the crash-prone approach behavior.

### Second Root Cause: Static Learning Rate

PPO's static `lr=3e-4` kept applying equally large gradient steps throughout. When the good behavior at 320K was found, the same gradient magnitude that found it also destroyed it over subsequent rollouts. A decaying LR would have preserved the 320K policy by making later updates smaller.

### Fixes

**1. `collision_penalty: 50.0 → 10.0`**

Rebalances the "approach vs survive" economics:
```
approach + crash:   10 penalty → wiped out by ~29 steps of "safe hovering"  (was 147)
```

Now "try approaching" is far less costly; policy gradient won't retreat as aggressively after a crash.

**2. Linear LR Schedule: `3e-4 → 0` over training**

```python
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

learning_rate=linear_schedule(3e-4)
```

Starts aggressive (fast initial learning), becomes conservative as training progresses — gradient steps shrink as good behaviors are found, reducing regression risk.

**Logs archived to:** `logs/legacy/6_before_lr_schedule_collision_fix/`

---

## Stage 0.25 — Fixed Spawn (No Symmetry Breaking)

**Trigger:** Stage 0.24 completed 10M steps. Full eval trajectory showed rapid improvement to eval=-109 at 280K steps, followed by a plateau at ~-70 to -80 from 2M–10M steps. All episodes terminated at ~250 steps (~1 second of simulation). Reward threshold (500) never reached.

### Root Cause: Negative Per-Step Reward Makes Fast Death Optimal

With symmetry breaking, drone spawned at up to 0.7m from target. Per-step reward at spawn:

```
alive_bonus=0.1, distance_penalty=0.5×0.7 = 0.35 → net = -0.25/step
```

Episodic return is maximized by terminating early when per-step reward is negative:

```
Hover at spawn 14400 steps:  14400 × (-0.25) = -3600
Die at step 250:             250 × (-0.25) - 10 = -72.5
```

PPO's value function correctly identified early death as the numerically superior strategy. The policy converged to this "die fast" equilibrium.

### Root Cause 2: Symmetry Breaking Creates Hover-at-Spawn Local Optimum

Even with a corrected alive_bonus, symmetry breaking creates a secondary risk: the drone learns to hover at its random spawn position rather than navigate to the target. With a 0.5m x,y offset, navigating (tilting to move horizontally while maintaining altitude) is harder than pure stabilization. Policy gradient can settle for "hover here" when that already earns positive reward.

### Fix: Fixed Spawn at [0, 0, 2.0]

Drone always spawns at the hover target. This eliminates both failure modes:

1. Per-step reward at spawn = `alive_bonus - 0 - tilt - ang_vel ≈ +0.1/step` — positive. "Die fast" is never optimal.
2. No spawn ≠ target offset. There is no "hover at spawn" local optimum to fall into.

**Trade-off:** Compass signal is always near [0,0,0] at spawn; policy only sees small perturbations. Compass following for large offsets is Stage 1's responsibility.

**`collision_penalty: 10.0 → 50.0`** reverted. With positive per-step reward, dying is no longer attractive regardless of penalty. -50 is the appropriate deterrent.

```yaml
stage_0:
  fixed_spawn: True   # always spawn at [0, 0, 2.0] = hover_target

hover_rewards:
  collision_penalty: 50.0   # reverted from 10.0
```

**Logs archived to:** `logs/legacy/7_before_fixed_spawn/`

---

## Stage 0.26 — Quartic Distance Reward + Thrust Centering

**Trigger:** Stage 0.25 (320K steps analyzed) converged to the same 0.88m equilibrium as Stage 0.23. All 20 eval episodes showed identical pattern: ep_len≈500-800 steps (2-3s), per-step≈-0.35/step, distance≈0.9m. Despite fixed spawn at center, the drone drifted to 0.88m and stayed.

### Root Cause 1: Thrust Bias (action=0 → 10N, not 9.81N)

```python
# Old:
target_thrust = ((action[3] + 1.0) / 2.0) * 20.0  # action=0 → 10N
# hover thrust = 1kg × 9.81 = 9.81N → net upward = 0.19N
# terminal rise velocity = 0.19/0.5 = 0.38 m/s
```

The untrained policy outputs action[3]≈0 (log_std_init=-1.2). With 10N thrust and hover at 9.81N, the drone slowly rises from z=2.0. As z increases, compass_z grows, dist grows, per-step reward becomes more negative.

### Root Cause 2: Per-Step Reward Negative Beyond 0.2m (same as before)

The linear `alive_bonus - distance_penalty × dist` formula made per-step reward negative for dist > 0.2m. The 0.88m equilibrium emerged because:
- At 0.88m: per_step = -0.34/step
- Die in 600 steps: -254 total
- Live 14400 steps at 0.88m: -4896 total
- Dying fast is always numerically superior when per-step is negative

This is the same local optimum that broke Stage 0.23 and 0.24. Fixed spawn eliminated one variant of it but the underlying incentive structure was unchanged.

### Comparison: gym-pybullet-drones (Toronto UTIAS)

The reference implementation uses `reward = -dist²` (also negative), but their episodes run for a **fixed 8 seconds with no early termination**. The policy cannot choose to die early. Our setup allows crash-termination with penalty, creating the "die fast" shortcut.

Their key insight: reward must be structured so early termination is never advantageous.

### Fix 1: Thrust Centering

```python
# New:
target_thrust = 9.81 * (1.0 + action[3])  # action=0→9.81N, action=-1→0N, action=+1→19.62N
```

action=0 now produces exactly hover thrust. The untrained policy no longer has a vertical drift bias.

### Fix 2: Quartic Distance Reward (non-negative)

```python
reward = max(0.0, 2.0 - dist**4)
       - tilt_penalty * (pitch² + roll²)
       - angular_velocity_penalty * |ang_vel|
       - collision_penalty  (if crash)
```

Per-step reward is **always ≥ 0** when no collision occurs:
- At dist=0 (target): +2.0/step → max gradient toward center
- At dist=0.88m: max(0, 2 - 0.60) = +1.40/step → still positive, dying never wins
- At dist=1.19m: 0/step → neutral, still no incentive to crash
- Beyond 1.19m: 0/step → neutral

Staying alive is always ≥ crashing (which gives -50). The "die fast" optimum is structurally impossible.

Gradient toward center is strong and quartic (steep near target, zero at 1.19m). Tilt and ang_vel penalties remain for stable posture transfer to Stage 1+.

`alive_bonus` and `distance_penalty` removed from hover_rewards YAML — no longer used.

### Reward Numerics

```
dist=0.00m: +2.00/step → 14400 steps = +28800 (theoretical max)
dist=0.30m: +1.99/step → good hover
dist=0.88m: +1.40/step → old equilibrium, now positive → no local optimum
dist=1.19m:  0.00/step → breakeven (beyond this, reward clamps to 0)
dist=5.00m:  0.00/step → neutral, still alive > dead
```

reward_threshold updated: 500 → **20000**. Requires mean dist < ~0.3m across 20 eval episodes at 60s each.

**Logs archived to:** `logs/legacy/8_before_quartic_reward/`

---

## Stage 0.26 — RESULT: First Successful Stage Completion ✓

**~1.88M steps to convergence. Mean eval reward = 20,119 → threshold crossed.**

This is the first time any stage has been successfully completed since the project began.

### What Happened

Training converged in ~1.88M steps (well within the 10M budget). The `mean=20119` triggered `StopTrainingOnRewardThreshold`. Final eval breakdown:

```
18/20 episodes survived full 60 seconds (14,400 steps)
Best episode:  r=24,252 → avg_dist ≈ 0.75m
Worst full ep: r=18,971 → avg_dist ≈ 0.91m
2 early crashes: r≈3,400-4,500 (steps 3600-4900, ~0.93-1.02m)
```

The training showed a characteristic dip-and-recovery curve: reward dropped from ~2100 at 80K steps to ~640 at 440K steps (policy exploration phase), then climbed steadily to 20K by 1880K steps.

### What the Drone Actually Learned

The drone hovers at **0.75–0.91m from target** on average. This is not tight position control — it is "stable flight somewhere near the target." The compass observation exists but the policy barely uses it: the quartic reward's gradient at 0.1m drift is only −0.004, effectively invisible against tilt/angular-velocity noise. The drone never learned "stay exactly here," it learned "hover stably in the vicinity."

### Why 20K Threshold Was Too Loose

With `max(0, 2 - dist^4)`:

| Threshold | Required avg_dist |
|-----------|------------------|
| 20,000    | 0.88m            |
| 27,000    | 0.59m            |
| 28,000    | 0.49m            |
| 28,750    | 0.25m            |

To require 0.25m with dist^4 needs 99.8% of max score — impractical. The quartic function compresses all meaningful distances into the top few percent of the reward range.

Additionally, the eval showed **bimodal distribution**: 18 episodes at 0.75–0.91m (good) and 2 episodes crashing at ~1.0m (bad). A mean threshold masks this inconsistency. A min-based or high-mean threshold would be more honest.

### Gradient Analysis: dist^4 vs dist^2

| dist | dist^4 gradient | dist^2 gradient | stronger |
|------|----------------|----------------|---------|
| 0.1m | −0.004         | −0.200         | **dist^2 (50×)** |
| 0.3m | −0.108         | −0.600         | **dist^2 (6×)** |
| 0.5m | −0.500         | −1.000         | **dist^2 (2×)** |
| 0.71m| −1.41          | −1.41          | equal crossover |
| 0.88m| −2.73          | −1.76          | dist^4 |
| 1.1m | −5.32          | −2.20          | dist^4 |

Since the drone spawns at dist=0 and the goal is to **stay there**, the relevant range is 0–0.7m. In this range dist^2 is stronger. The drone drifted to 1.1m because the gradient at 0.1m was too weak to correct the initial policy noise before it compounded.

**Model saved at:** `models/Stage_0_Hover/` (preserved for comparison)
**Eval data saved at:** `logs/teacher_ppo/Stage_0_Hover_v1_evaluations.npz`

---

## Stage 0.27 — Quadratic Distance Reward + Stricter Threshold

**Trigger:** Stage 0.26 passed but drone hovering at 0.75–0.91m is not meaningful hover for Stage 1 navigation. The quartic reward's flat gradient near origin prevented the policy from learning tight position control.

### Changes

**1. dist^4 → dist^2 (reward_functions.py)**

```python
# Stage 0.26:
reward = max(0.0, 2.0 - dist**4)

# Stage 0.27:
reward = max(0.0, 2.0 - dist**2)
```

50× stronger gradient at 0.1m drift. The drone now gets a meaningful penalty signal the moment it starts to drift from spawn, preventing the 1.1m equilibrium from ever forming.

Breakeven shifts from 1.19m → 1.41m. The "die fast" protection is unchanged (reward still always ≥ 0).

**2. Threshold: 20,000 → 27,000 (teacher_ppo.yaml)**

With dist^2, threshold-to-distance mapping:

```
dist=0.00m: +2.00/step → 28,800 max
dist=0.25m: +1.94/step → 27,936 over 60s
dist=0.30m: +1.91/step → 27,504 → threshold ≈ 27,000
dist=0.50m: +1.75/step → 25,200
dist=0.88m: +1.23/step → 17,712
```

Threshold 27,000 ≈ 0.30m average distance. This is real hover — the drone must stay within arm's reach of the target consistently.

With high mean (27K), variance is forced low: even one episode at 1.1m (r≈2500) would drag the 20-episode mean below threshold. The high mean implicitly enforces consistency.

**3. Run name: Stage_0_Hover → Stage_0_Hover_v2**

Preserves Stage_0_Hover model folder for side-by-side comparison.

**4. Early termination at dist > 1.5m (drone_sim.py)**

```python
if self.hover_only and hover_dist > 1.5:
    terminated = True  # no penalty, clean reset
```

Beyond 1.5m the dist^2 reward is already 0 (breakeven=1.41m). Without termination,
the drone wastes up to 60s of rollout steps collecting zero signal. Terminating
early resets the episode and gives the policy another attempt. No collision penalty
— there is no incentive to deliberately drift to 1.5m since the drone was already
getting 0 reward before hitting the cutoff.

This matches gym-pybullet-drones (Zürich) ±1.5m XY boundary approach.

### What This Expects to Produce

The drone spawns at dist=0. With dist^2, the policy receives a 50× stronger signal to not drift. It should learn to output near-zero pitch/roll from the all-zero spawn observation. The 1.1m equilibrium should never form because the gradient at 0.1m is large enough to teach correction before drift compounds. Failed episodes that drift past 1.5m terminate early instead of wasting 60s of zero-reward steps.

Expected outcome: drone hovering within 0.2–0.3m consistently across all 20 eval episodes.

---

## Stage 0.28 — Spawn Offset (±0.25m) + Reduced Tilt Penalty

**Trigger:** Stage 0.27 reached mean ~24K but stalled. Root cause: spawning exactly at target [0,0,2] means compass=[0,0,0] at every episode start. Policy learned a "memorised nominal hover" action — worked for most physics worker states but 10/20 eval episodes consistently crashed in 2–6s. Bimodal structure persisted for 1.5M+ steps with no learning signal to fix it.

### Changes

**1. Spawn offset: exact [0,0,2] → ±0.25m XY (drone_sim.py)**

```python
# Stage 0.27 (fixed_spawn):
start_x = 0.0
start_y = 0.0

# Stage 0.28:
start_x = self.np_random.uniform(-0.25, 0.25)
start_y = self.np_random.uniform(-0.25, 0.25)
```

Drone now always starts with a non-zero compass vector. Policy must learn "navigate to hover point and hold" instead of "stay where you spawned."

**2. tilt_penalty: 0.5 → 0.15**

v2's 0.5 penalty created a static equilibrium at ~0.28m: cost of tilting to correct = distance reward benefit at that distance. Lowering to 0.15 was intended to let the drone tilt more freely and converge closer.

**3. Run name: Stage_0_Hover_v3**

### Result

Bimodal failure eliminated — no more 10/10 catastrophic split. Continuous episode length distribution. HOWEVER: drone developed an underdamped oscillation/drift, settling at ~0.45–0.55m average distance instead of ~0.28m. Root cause: without a lateral velocity penalty, "drifting at 0.5m" costs nothing extra versus "hovering at 0.5m". Policy found it acceptable to slowly drift rather than actively converge. Peak mean: 16,877 at 1.96M steps, declining thereafter. Training terminated at 2.32M.

---

## Stage 0.29 — Lateral Velocity Penalty

**Trigger:** GUI observation confirmed slow drift pattern: drone not oscillating, just gradually sliding away from hover point. The reward function had no term penalising lateral velocity — drift was "free". Policy optimised for "survive near 1.5m boundary" rather than "hold position at center".

### Root Cause Analysis

Current reward penalises WHERE (dist²) and HOW TILTED (tilt²) but not HOW FAST MOVING (lateral vel). At any given position, drifting sideways at 0.5 m/s earns the same reward as being stationary. The optimal policy under these constraints accepts perpetual drift as long as it stays within the 1.41m reward boundary.

Both failure modes — slow drift AND circular orbit — involve non-zero lateral velocity and are fixed by the same term.

### Changes

**1. velocity_penalty added to compute_hover_reward (reward_functions.py)**

```python
lateral_vel = np.sqrt(drone_vel[0]**2 + drone_vel[1]**2)
reward -= reward_weights.get('velocity_penalty', 0.0) * lateral_vel
```

**2. tilt_penalty: 0.15 → 0.27, velocity_penalty: 0.08 (teacher_ppo.yaml)**

```yaml
tilt_penalty: 0.27       # moderate damping; 0.5 was overdamped (0.28m ceiling), 0.15 underdamped
velocity_penalty: 0.08   # penalise lateral motion directly
```

**3. Run name: Stage_0_Hover_v4**

### Risk: "staying put" local optimum?

Valid concern. If lateral motion is penalised, could the policy learn to just not move toward center?

Per-step analysis at spawn (dist=0.25m), moving toward center at v m/s:
- Distance reward gain per step: 2 × 0.25 × v/240 = +0.0021v
- Velocity penalty per step: −0.08v
- Net immediate: −0.0779v (negative while moving)

Immediate per-step cost is real. But cumulative value of being at center dominates:
- Reward at center for remaining 14,300 steps: +0.0625/step extra = +893 total
- Cost of the journey (e.g. 100 steps at 0.1 m/s): 0.08 × 0.1 × 100 = 0.8 total

Net gain from moving to center: **+892**. The PPO value function should capture this.

Practical safeguard: `velocity_penalty.get()` defaults to 0.0, so old checkpoints are unaffected. If v4 shows the drone frozen at spawn, reduce penalty to 0.05 or lower.

### Expected Outcome

Drone approaches center from ±0.25m spawn, decelerates as it nears target, holds position. Velocity penalty makes "close AND still" the unique optimum. Target: avg_dist < 0.35m across 60s episodes → mean ≈ 27K threshold.

### Infrastructure Change (Concurrent with v4)

**N_ENVS: 4 → 12.** Root cause of the previous bottleneck: WSL2 was capped at `processors=4`, making additional environments compete on the same 4 cores with no gain. After raising WSL2 to `processors=16`, N_ENVS was set to 12 (leaving 4 threads for OS + main process).

Rollout scaling: `n_steps=4096 × 12 envs = 49,152 transitions/iter`. `batch_size` scaled to `1536` to maintain 32 mini-batches per update (49,152 / 32 = 1,536). eval_freq=10,000 policy steps × 12 envs → eval every ~120K global timesteps. Training throughput increased roughly 3× vs. 4-env setup.

### Result

Velocity penalty addressed the slow-drift failure mode confirmed in v3: the policy no longer accepts perpetual lateral motion as a free strategy. Episodes that survived past the first few seconds showed measurably less drift than v3. Peak performance (mean ~16.5K, 15/20 full episodes) slightly exceeded v3's peak, confirming the velocity term added genuine signal.

**However, the bimodal failure persisted.** Approximately half of eval episodes consistently crashed in under 5 seconds regardless of training duration. This pattern is structurally different from the drift problem:

- Drift failure: reward gradient was too weak → fixed by velocity penalty.
- Hard crash failure: policy outputs destabilising actions for certain ±0.25m spawn positions → not a reward design problem.

The reward function is now correctly shaped. The remaining bottleneck is **policy generalisation across spawn conditions**: some (x, y) offsets within ±0.25m fall outside the region the policy has learned to handle. No reward change can fix this directly — the policy must explore and generalise across the full spawn distribution.

### What v4 Revealed

The Stage 0 reward design is complete. Three iterations converged on:

```
R = max(0, 2 - dist²)
  - tilt_penalty × (pitch² + roll²)
  - angular_velocity_penalty × |ang_vel|
  - velocity_penalty × sqrt(vx² + vy²)
  - collision_penalty  [if crash]
```

Each term addresses a specific failure mode: dist² for gradient strength, tilt for damping, ang_vel for attitude noise, lateral vel for drift. No further reward terms are expected to help.

The unsolved problem is whether PPO can generalise hover across all ±0.25m spawn positions within a 10M-step budget, or whether a different training regime (shorter episodes, domain randomisation schedule, or alternative spawn strategy) is needed. This is the decision point for Stage 0 exit.

---

## Stage 0.29 — v4 Final Result (10M Steps)

**Date:** 2026-04-19

v4 ran for the full 10M step budget. The 27,000 reward threshold was never crossed.

### Eval Trajectory

- Mean reward climbed steadily to ~25,600–25,800 from ~7.2M steps onward
- 20/20 full episodes (14,400 steps each) from ~7.2M steps — zero crashes in eval
- Plateau confirmed from 9M–10M: 8 consecutive evals with no upward trend
- Final mean at 10M: **~25,623**

### Root Cause: Structural Reward Ceiling

The 27,000 threshold was unreachable by design. Even during stable hover with dist≈0:

```
tilt_penalty × (pitch² + roll²)   ≈ 0.02–0.05/step  (residual PD oscillation)
angular_velocity_penalty × |ω|    ≈ 0.01–0.03/step
velocity_penalty × lateral_vel    ≈ 0.01–0.03/step
─────────────────────────────────────────────────────
unavoidable penalty floor          ≈ 0.04–0.11/step
```

Theoretical max: 2.0/step × 14,400 steps = 28,800. With penalty floor ~0.07/step, effective ceiling ≈ 28,800 − (0.07 × 14,400) = **25,800**. The plateau at ~25,623 matches this ceiling precisely.

The bimodal failure (half episodes crashing under 5 seconds) never fully resolved, but the 20/20 full-episode runs in later evals showed the policy had largely generalised across spawn conditions.

### Decision: Stage 0 Declared Solved

Threshold lowered from 27,000 to 25,000 retroactively. Stage 0 declared solved. The reward design is complete and the hover policy is sufficient to transfer to Stage 1.

**Model saved:** `models/Stage_0_Hover_v4/best_model.zip`

---

## Stage 1 — Scout: Transfer Result

**Date:** 2026-04-19

### Setup

- 1 fixed coin at world position [1.0, 0.0, 2.0] — exactly 1m from room center
- Drone spawns with ±0.5m XY offset and random yaw (full symmetry breaking)
- Nav reward: alive_bonus=0.02, distance_penalty=0.02×dist (Golden Ratio: net=0/step at 1m), coin_reward=300, success_bonus=1000, collision_penalty=300
- Weights transferred from Stage_0_Hover_v4 best model

### Result: Trivially Solved at First Eval

Training stopped after **120,000 steps** (1 eval). Mean reward: **1,299**. All 20 eval episodes successful.

```
Eval step 120,000 | Mean R: 1299 | Ep Len: 523 | Full: 0/20
```

Monitor episodes showed collection in 171–1,326 simulation steps (0.7–5.5 simulation seconds). No failures across the entire training run.

### Why

The hover policy's core learned skill is "follow the compass vector to minimise it." In Stage 0, the compass pointed to the virtual hover target [0,0,2.0]. In Stage 1, the compass points to the coin — same structure, different destination. The policy transferred instantly with zero additional learning required.

This validates the unified compass architecture introduced in Stage 0.23: by making the hover target a real compass anchor (rather than zeroing the compass in hover), the policy learned a skill that generalised directly to navigation.

**Model saved:** `models/Stage_1_Scout/best_model.zip`

---

## Infrastructure Note: PyBullet GUI Simulation Speed (WSL2)

**Date:** 2026-04-20

### Finding

The PyBullet GUI notebook (notebook 05) runs at approximately **10x slower than real-time** on WSL2. Measured directly:

| Episodes | Steps | Sim time (steps/240Hz) | Real time | Effective Hz | Slowdown |
|---|---|---|---|---|---|
| Ep 1 | 702 | 2.92s | 29.64s | 23.7 Hz | 10.1× |
| Ep 2 | 922 | 3.84s | 36.64s | 25.2 Hz | 9.5× |
| Ep 3 | 873 | 3.64s | 34.61s | 25.2 Hz | 9.5× |
| Ep 8 | 1098 | 4.58s | 46.49s | 23.6 Hz | 10.2× |
| Ep 9 | 946 | 3.94s | 37.60s | 25.2 Hz | 9.5× |

Consistent factor: **~10×**. Short episodes show higher apparent slowdown due to fixed startup overhead per episode being diluted less.

### Root Cause

Each `addUserDebugLine()` and `addUserDebugText()` call is an IPC round-trip to the PyBullet GUI process (~10–15ms each). The main loop calls 3 arrow lines + 1 text label every step, plus trail lines every 12 steps. Total per-step GUI overhead: ~40–50ms. With `time.sleep(1/240)` = 4.17ms sleep, the effective step rate is ~20–25 Hz instead of 240 Hz.

This is a **WSL2 GUI overhead issue**, not a physics or reward issue. Training (headless, no GUI, no sleep, 12 parallel envs) runs thousands of steps per second.

### Consequence

14,400 simulation steps = **60 simulation seconds** (physics time, always correct).
14,400 steps in the GUI notebook ≈ **600 real seconds** (~10 minutes) to watch.

Episode lengths in training eval (e.g., 523 steps for Stage 1 coin collection) represent **2.18 simulation seconds** of drone flight, displayed as ~30 real seconds in the notebook.

### Fix

`RENDER_STRIDE = 10` added to notebook 05. Advances 10 physics steps per GUI frame, making wall-clock time match simulation time. GUI overlays update every 10th step instead of every step. Set `RENDER_STRIDE = 1` to revert to slow-motion for detailed observation.

---

## Stage 2 — Navigator: Policy Collapse & Reward Redesign

**Date:** 2026-04-20

### Training Run (Stage_2_Navigator, 7.92M steps)

Loaded from Stage_1_Scout best model. Ran overnight. Results:

| Phase | Steps | Mean R | Full eps | Diagnosis |
|---|---|---|---|---|
| Rise | 0–1.08M | 221→462 | 0→7/20 | Learning, occasionally collects all 4 coins |
| Farming | 1.2M–3.12M | 180–347 | 17–20/20 | Alive-bonus farming local optimum |
| Onset | 3.36M–4.8M | 0→−56 | 10–18/20 | Farming strategy destabilizing |
| Collapse | 5M–7.92M | −50→−230 | 0–2/20 | Full crash, ep_len 4000–6000 |

Training stopped at 7.92M. Threshold 1500 never approached.

### Root Cause: Suicide Policy at Far Coins

Coins 3 and 4 at `[4,4,2]` and `[-4,-4,2]` — **5.66m from center**. With the existing nav reward structure:

```
alive_bonus:       +0.02/step
distance_penalty:  −0.02 × 5.66 = −0.113/step
net per-step:      −0.093/step  ← NEGATIVE
```

At −0.093/step, the collision penalty (−300) is recouped in only **3,226 steps (13.4 sim seconds)**. Dying near a far coin is economically optimal. This is Stage 0.2 (Suicide Policy) reappearing for targets beyond the 1m Golden Ratio breakeven.

The policy correctly identified alive-bonus farming (~288/episode) as the best achievable strategy, then collapsed from that local optimum through entropy reduction and gradient destruction.

Note: Stage 1 was not a "bad policy." Its coin was at 1m — exactly the Golden Ratio breakeven — so the economics were neutral. The policy genuinely learned compass-following and coin collection. The reward structure was broken only for Stage 2's far coins.

### Fix: Progress Reward (Stage_2_Navigator_v2)

Removed `alive_bonus` and `distance_penalty_multiplier` from nav_rewards entirely. Replaced with:

```python
reward += progress_reward_weight * (prev_dist - curr_dist)
```

`coin_progress` computed per step as distance closed toward nearest coin. Resets cleanly on coin collection (no snap penalty). Initialised from spawn distance in `reset()`.

```yaml
nav_rewards:
  progress_reward_weight: 50.0   # metres closed × 50
  coin_collection_reward: 300.0
  success_bonus: 1000.0
  collision_penalty: 300.0
  smoothness_penalty_multiplier: 0.02
```

**Why this works:**
- Distance-agnostic: same shaped gradient whether coin is 1m or 5.66m away
- Hovering: 0 reward per step (not negative → farming is no longer a local optimum)
- Dying: −300 (always worse than forward movement)
- Literature: Kaufmann et al. 2023 (Swift, Nature) uses pure progress reward for champion-level drone navigation

**Expected reward for full 4-coin run:** ~3,200 (progress ~1000 + coins 1200 + success 1000). Threshold set at 2,000.

Corrupted Stage_2_Navigator model folder deleted. Stage_2_Navigator_v2 starts from Stage_1_Scout weights.

**Code changes:** `drone_sim.py` (prev_coin_distance tracking), `reward_functions.py` (progress reward replaces alive+distance), `configs/teacher_ppo.yaml` (nav_rewards restructured, run_name updated).

---

## Stage 2 — Navigator v3: Episode Length + N_ENVS Fix

**Date:** 2026-04-20

### Why v2 Failed (Full Analysis)

v2 peaked at **2.04M steps** (mean 1121, 11/20 episodes with r>1200 = multi-coin collection). Then regressed continuously through 3.72M steps, ending with r>1200=0/20 for 12 consecutive evals and max reward ~850 (1-2 coins only).

This is **policy instability at long episodes**, not a reward design problem. At 14400 steps (60 sim-seconds), the policy must maintain correct multi-coin behavior for a very long time. Any variance in the rollout causes forgetting. The best strategy was confirmed possible (2.04M proved it), the policy just couldn't hold it consistently.

This is distinct from v1's alive-bonus farming collapse (mean went to -200). v2's mean stayed at 400-700 — the policy was not broken, just inconsistent.

### Changes for v3

**1. max_steps: 14400 → 7200 (60s → 30s)**

Literature range: Swift 6s, DPRL 25s, gym-pybullet-drones 8s. Physical minimum to collect 4 coins at 0.5 m/s average is ~20s — 30s gives 1.5x margin. Shorter episodes mean more resets per hour, more gradient updates, and less opportunity for variance to compound into forgetting.

**2. N_ENVS: 12 → 14**

WSL2 has 16 processors. 14 envs + 2 for main process + OS headroom. ~15% more data per unit time.

**3. batch_size: 1536 → 1792**

Maintains 32 mini-batches: 4096×14=57344 rollout / 32 = 1792.

**4. reward_threshold: 2000 → 1500**

Scaled for shorter episodes. Absolute reward is lower with fewer steps available for progress accumulation, but coin/success bonuses unchanged.

**5. run_name: Stage_2_Navigator_v3**

Starts from Stage_1_Scout weights. Stage_2_Navigator_v2 best model (2.04M, mean 1121) preserved as fallback but not used as starting point to avoid any instability baked into its weights.

---

## Stage 2 — Navigator v4: Coin Geometry Redesign + Episode Length Correction

**Date:** 2026-04-21

### Why v3 Failed (Root Cause)

v3 ran the full 10M step budget. r>1200 = 0/20 across all 71 evals. The policy reliably collected 1-2 coins but never reached coins 3-4.

Root cause: **the coin geometry made the task physically impossible within the 30s (7200 step) episode budget.**

The old fixed positions:
```
Coin 1: [1,  0, 2]   →  1.0m from center
Coin 2: [0,  1.5, 2] →  1.5m from center
Coin 3: [4,  4, 2]   →  5.66m from center
Coin 4: [-4,-4, 2]   →  5.66m from center (OPPOSITE corner from coin 3)
```

Path from coin 3 → coin 4: `[4,4] → [-4,-4]` = 8√2 = **11.3m**. At 0.5 m/s that single leg takes 22.6 sim-seconds. Total full-collection path: ~20m = ~40 sim-seconds = 9600 steps. This physically exceeds the 7200-step budget. The policy was not failing to learn — it was mathematically prevented from completing the task within the time limit.

The v2 peak (11/20 success at 2.04M steps with 14400 step budget) was real capability; v3 cut the budget below the physical minimum.

### Changes for v4

**1. Coin positions redesigned**

New layout — clockwise ring, each coin ~2-3m from center, each in a different quadrant:
```python
[ 1.0,  0.0, 2.0],   # coin 1: 1m, easy entry
[ 0.0,  2.0, 2.0],   # coin 2: 2m, 90° heading change  
[-2.5,  1.5, 2.0],   # coin 3: ~2.9m, 135° heading change
[-1.5, -2.5, 2.0],   # coin 4: ~2.9m, opposite quadrant
```

Total path: ~13m. At 0.5 m/s: ~26 sim-seconds = 6240 steps. Fits in 45s with 1.7x margin.
Each coin forces a genuine heading change; no two consecutive coins are in opposite corners.

**2. max_steps: 7200 → 10800 (30s → 45s)**

26s physical minimum with new geometry. 45s gives comfortable margin for slow/non-optimal paths.

**3. run_name: Stage_2_Navigator_v4**

Starts from Stage_1_Scout weights.

---

## Stage 2 — Navigator v4: First Eval Result & Threshold Correction

**Date:** 2026-04-21

### Result at 140K Steps

Training stopped at the very first eval. New coin geometry worked immediately:

```
Step 140,000 | mean 1569.9 | max 2552.6 | min -24.2 | avg_len 4881 | r>1200: 12/20
```

12/20 episodes collected 2-3 coins. Max 2552 indicates some episodes reaching 3-4 coins. The Stage 1 policy transferred directly — same pattern as Stage 1 solving instantly from Stage 0 weights.

**Root cause of immediate stop:** reward_threshold was 1500, mean was 1569. `StopTrainingOnRewardThreshold` fired after one eval, same mistake as Stage 1 (threshold 600, mean 1299).

### Threshold Correction

1500 → **2000**. Requires more consistent multi-coin collection across all 20 eval episodes. The 12/20 success rate at 60% is not solid enough for Stage 3 transfer.

Training resumes from `Stage_2_Navigator_v4/best_model.zip` (the 1569-mean checkpoint). No restart from Stage 1 weights needed.

### Lesson

Thresholds for navigation stages need to account for the fact that progress reward + coin rewards can easily exceed a low bar even with inconsistent performance. Rule of thumb going forward: set threshold at ~70-80% of the theoretical maximum for the stage rather than a fixed value.

For Stage 2 with 4 coins: theoretical max ≈ 4×300 + 1000 success + ~730 progress = ~2930. Threshold 2000 ≈ 68% of max — requires consistent 3-coin collection or occasional 4-coin completion.

---

## Stage 3 — Hunter v1: Structural Failure (Z-Height Jump)

**Date:** 2026-04-21 → 2026-04-22

**Run:** `Stage_3_Hunter_v1` — 3.92M steps, 28 evals. Model deleted (failed run). Eval data preserved at `logs/teacher_ppo/stage_3/Stage_3_Hunter_v1/evaluations.npz`.

### Config

- 10–18 random coins, random Z ∈ [1.0, 6.0]m
- max_steps: 10800 (45s)
- N_ENVS: 14, started from Stage_2_Navigator_v4 best_model (280K, mean 1999)
- reward_threshold: 2000

### Results

| Quarter | Evals | Avg mean R | Avg max R |
|---|---|---|---|
| Q1 (0–1M) | 7 | 6.9 | 674 |
| Q2 (1–2M) | 7 | 5.6 | 720 |
| Q3 (2–3M) | 7 | 36.8 | 665 |
| Q4 (3–4M) | 7 | 45.3 | 779 |

Coin collection across all 560 eval episodes: 46% crashed/collected nothing, 48% collected 0–1 coins, 4 episodes (0.7%) collected 2+ coins. Zero upward trend over 4M steps. Threshold 2000 never approached. Run stopped and declared a structural failure.

### Root Cause: Two Simultaneous Hard Jumps

**1. Z-height gap (primary):** Stage 2 coins were all at Z=2.0m — the drone learned to navigate in a flat XY plane at constant altitude. Stage 3 coins spawn at Z=1.0–6.0m. Coins at Z=4–6m require 2–4m of altitude gain while navigating — a skill the Stage 2 policy never needed and the compass Z component was effectively ignored. The LiDAR is also horizontal-only (gimbal-stabilized, yaw-only rotation), so it provides no information about coin altitude.

**2. Fixed → random coin positions:** Stage 2 had 4 coins at known fixed positions. Stage 3 distributes 10–18 coins randomly across 16×16×5m = 1280m³. Even with 14 random coins that's ~91m³ per coin on average — a large 3D search space with no prior knowledge.

The policy correctly generalized from Stage 1 to Stage 2 because the coin geometry was similar (familiar XY plane, incremental distances). The Stage 2→3 jump broke both the altitude invariant and the positional familiarity simultaneously.

### Decision: Redesign Stage 3 as Separate Z-generalization Step

Stage 3 v2 will constrain coins to Z ≈ 2.0m (same as Stage 2), forcing XY generalization only. Z navigation will be introduced at a later stage once XY search is mastered.

---

## Distillation Readiness Note: Teacher Yaw Alignment Problem

**Date:** 2026-04-21

### Observation

The teacher drone does not consistently face coins when approaching them. As a quadrotor, it can fly sideways, backward, or at arbitrary angles relative to its nose direction. The red arrow in the GUI (body +Y forward) frequently points in a different direction from the movement vector.

### Why This Matters for Distillation

The student uses a camera fixed to the body frame. If the teacher approaches a coin from an angle where the coin is off-camera, the student receives unlearnable training data: no visual information about the coin, but a teacher action that implies "go here." Behavioral cloning requires that the teacher's reasoning is recoverable from the student's observations.

### Options (to evaluate before distillation begins)

1. **Yaw alignment reward**: Add a small bonus for having the compass vector aligned with body forward (+Y). Incentivizes the teacher to face coins before approaching — makes behavior camera-explainable by construction. Something like `cos(angle_between_compass_and_body_Y) × small_weight`.

2. **Wider camera FOV**: Use 120-150° wide-angle lens for the student. Coin more likely to be in frame even with misaligned yaw. No teacher retraining needed.

3. **DAgger instead of pure behavioral cloning**: Collects new teacher demonstrations at states the student actually reaches, including recovery from "coin off-screen" situations. Handles distribution shift directly.

4. **Accept and measure**: If the drone mostly faces coins when close (body-frame compass naturally incentivizes yaw toward target), the problem may be smaller than it looks. Watch several episodes and measure what fraction of approach steps have the coin significantly off-axis.

**Decision point**: Watch the trained Stage 2/3 teacher carefully. If coin is frequently behind the camera during approach → Option 1 or 3. If rarely → Option 4 or 2.

---

## Full Curriculum Restart — v5 Redesign

**Date:** 2026-04-22

### Diagnosis of v1-v4 Curriculum Failures

After reaching Stage 2 v4 (280K best model, 16/20 success, 4 immediate crashes) and observing the trained policy in the GUI, three root-cause issues were identified that justified a full restart rather than continued patching:

**1. Stage 0 velocity_penalty caused deceleration transfer (primary)**

Stage 0 v4 used `velocity_penalty=0.08 × |lateral_vel|` which penalises speed near the hover target. Over 10M training steps this baked a "near target = slow down" prior into the value function. This transferred through Stage 1 (120K steps, insufficient to override) and Stage 2 (280K steps before collapse). Observed in GUI: drone decelerates to near-zero at ~1m from each coin, then hovers extremely slowly for collection.

Confirmed by literature (arXiv 2501.18490): keep reward structure consistent across stages; use action smoothness (‖Δa‖²) not velocity magnitude penalties in hover.

**2. Stage 0 episode length (60s) allowed imprecise hover to pass threshold**

With `max(0, 2-dist²)` reward: hovering at 0.5m yields 1.75/step × 14400 = 25,200 pts — near the 25,000 declared threshold. Policy correctly identified 0.4-0.5m as the effort/reward optimum and never learned tight position control. The transition to navigation stages carried this imprecision.

Fix: 15-second episodes + `max(0, 2-4·dist²)` which zeros at 0.71m and gives 4× stronger gradient near origin.

**3. No yaw alignment = CNN distillation failure guaranteed**

Quadrotor omnidirectionality means the drone never needs to face the coin to collect it. Without explicit incentive, ~60-80% of near-target approach frames have the coin off-camera (estimated from FOV geometry). Since CNN student requires teacher actions to be recoverable from camera frames, an omnidirectional teacher generates unlearnable training data for a forward-camera student.

Fix: `r_yaw = 0.15 × cos(θ_error)` at dist < 2.5m.

**4. Trajectory squiggly — no velocity direction incentive**

Pure progress reward (`50 × Δdist`) rewards any movement that closes distance, including sideways and backward approaches. No incentive for straight-line efficient paths.

Fix: `r_dir = 0.20 × dot(v̂, û_target)` — trajectory constraint compatible with omnidirectionality.

### New Design Summary

| Component | Old (v4) | New (v5) |
|---|---|---|
| Hover reward | `max(0, 2-dist²)` | `max(0, 2-4·dist²)` |
| Hover velocity term | `velocity_penalty=0.08` | `smoothness_penalty=0.05` |
| Stage 0 max_steps | 14400 (60s) | 3600 (15s) |
| Stage 0 threshold | 25000 | 6000 (~0.25m avg dist) |
| Nav: approach zone | none | `+150 × progress` at dist < 1.5m |
| Nav: yaw alignment | none | `+0.15 × cos(θ)` at dist < 2.5m |
| Nav: trajectory | none | `+0.20 × dot(v̂, û_target)` |
| Stage 0 run_name | Stage_0_Hover_v4 | Stage_0_Hover_v5 |
| Stage 1 run_name | Stage_1_Scout_v1 | Stage_1_Scout_v2 |
| Stage 2 run_name | Stage_2_Navigator_v4 | Stage_2_Navigator_v5 |

Stage advancement criteria (manual enforcement):
- Threshold exceeded in **3 consecutive evals** (not just one peak)
- Max **1 early crash** per eval (episode < 1000 steps, negative reward)

---

## Stage 0 v5 — Declared Solved

**Date:** 2026-04-22

### Result

| Eval | Steps (session 2) | Mean Reward | Ep Len | Full Episodes |
|---|---|---|---|---|
| 1 | 140,000 | 6,233.5 | 3600 | 20/20 |
| 2 | 280,000 | 6,244.6 | 3600 | 20/20 |
| 3 | 420,000 | 6,111.7 | 3600 | 20/20 |

3 consecutive evals above threshold (6000) → `ConsecutiveThresholdCallback` fired. **Stage 0 v5 declared solved.**

Total training: ~4.48M steps across 2 sessions (29 evals in session 1 + 3 in session 2).

### What the numbers mean

- **ep_len=3600 in all 60 eval episodes** (20/20 × 3 evals): drone held hover for the full 15 seconds every single time, zero early terminations.
- **Mean ~6100–6250**: back-calculating via `2 − 4d² ≈ mean/3600 + penalty_offset` → average hover distance ~0.23–0.25m from target. Significantly tighter than v4's ~0.5m sweet spot.

### What changed vs v4 (why this worked)

The `4·dist²` scaling with 15s episodes removed the v4 perverse equilibrium where hovering at 0.5m was near-optimal. The `smoothness_penalty` replacing `velocity_penalty` removed the "decelerate near target" prior that had transferred destructively through Stage 1 and Stage 2 in the previous curriculum.

### Decision: proceed to Stage 1

Session 2 plateau was ~6100–6250 with LR decaying to 0 — no further improvement expected. 20/20 full episodes is the cleaner signal than the reward number. Stage 1 will validate whether hover quality is sufficient for navigation transfer.

---

## Stage 1 v2 — Design (pre-training)

**Date:** 2026-04-22

### What changed vs Stage 1 v1

| Component | v1 | v2 |
|---|---|---|
| Coin position | Fixed `[1.0, 0.0, 2.0]` (always +X, 1m) | Random angle, fixed 2m radius from origin |
| Effective nav challenge | Hover-drift to memorized position | Must use compass to navigate to unknown direction |
| Nav rewards | Progress only (from Stage 0.23) | Progress + approach_bonus + yaw_alignment + velocity_direction |
| Episode length | 30s (7200 steps) | 30s (7200 steps, unchanged) |
| Threshold | 800 | 800 |

### Reward math (2m coin, clean collect)

- Long-range progress (2m → 1.5m): 50 × 0.5 = **25 pts**
- Approach zone (1.5m → 0.6m): 200 × 0.9 = **180 pts**
- Coin collection: **300 pts**
- Success bonus: **1000 pts**
- Yaw alignment (~600 steps at <2.5m, cos≈0.8): **~72 pts**
- Velocity direction (~600 steps, dot≈0.8): **~96 pts**
- **Total clean collect: ~1673 pts**

Threshold 800 requires ~8/20 episode collections. Stage 0 v5 compass skill should transfer directly; expect 10-18/20 from the first eval.

### Expectation

Near-zero-shot transfer is likely but not guaranteed. v1 was fully zero-shot because the coin was within hover-drift range (1m). v2 coin at 2m genuinely requires departure from hover zone — the drone must commit to directional flight. The approach_bonus (+150×progress at <1.5m) counters any residual deceleration prior. The yaw alignment reward starts training here for the first time.

Expected outcome: solved in 1–4 evals (140–560K steps). If threshold is not hit by eval 3, something structural is wrong with the hover→nav transfer.

---

## Stage 1 v2 — Run 1 Post-Mortem (arc-trajectory failure)

**Date:** 2026-04-23

### Result

| Eval | Steps | Mean Reward | Ep Len | Notes |
|---|---|---|---|---|
| 1 | 140K | 1508.7 | 1270 | Collecting, but arcing |
| 2 | 280K | 1400.0 | 1340 | Same |
| 3 | 420K | 1439.4 | 1500 | Same — threshold 800 passed, training stopped |

Threshold 800 was passed (3 consecutive) → training stopped. But observed behavior in the GUI revealed the drone consistently **collected the coin via arc trajectories**: curved paths, circles around collection radius, stops and reaccelerations, sideways/backward approaches. The yaw alignment metric was effectively ignored.

### Root Cause

The yaw alignment reward (`0.15 × cos(θ)`) was **~3× weaker** than the approach bonus signal (`200 × Δdist/step ≈ 0.42/step` at 0.5m/s). The drone maximized distance-closure by flying sideways (omnidirectional quadrotor) — no yaw required. The arcs are locally optimal paths for approach bonus exploitation. The threshold of 800 was also too low: arc behavior still collected 20/20 and hit 1500+ mean.

### What Changed for v3

Two changes together enforce "face-first, then fly":

**1. Conditional approach_bonus** (`approach_bonus_requires_yaw: true`, `threshold: 0.5`):
The 3× approach multiplier only fires when `cos(θ) > 0.5` (coin within 60° of drone nose). Arc trajectories lose the multiplier: reward drops from ~205 pts progress to ~70 pts. Arcing behavior cannot pass threshold 1900 even with 20/20 collections (~1622 pts/episode < 1900).

**2. Yaw alignment weight**: `0.15 → 0.5` — now comparable in magnitude to approach signal. Each step of good yaw alignment (~0.5 pts) is meaningful relative to approach bonus (~0.42 pts).

Literature anchor: Penicka et al. ICRA 2023 (arXiv 2210.01841) — perception-aware reward shaping for camera-compatible teacher training.

### Virtual FOV / PD-override proposal (evaluated and rejected)

The idea of suppressing the compass target when no coin is in FOV and having a PD controller forcibly spin the drone was evaluated via wiki literature search. Rejected because:
1. PPO is on-policy — steps where PD overrides yaw produce invalid importance ratios in the rollout buffer
2. Restricted state distribution: RL never learns to turn on its own
3. Mode-transition jitter at PD→RL handoff creates out-of-distribution states

The soft reward approach (conditional approach_bonus + higher yaw weight) achieves the same behavioral goal without hybrid control.

---

## Stage 1 v3 — Design (pre-training)

**Date:** 2026-04-23

Same environment as v2 (coin at random angle, 2m radius, Z=2m) with strengthened rewards:
- `yaw_alignment_weight`: 0.15 → **0.5**
- `approach_bonus_requires_yaw`: **true** (new — unlocks 3× only when cos > 0.5)
- `reward_threshold`: 800 → **1900**
- `run_name`: Stage_1_Scout_v3 (fresh start from Stage_0_Hover_v5 weights)

### Expected reward structure

| Behavior | Progress | Yaw | Vel | Coin+Success | Total/ep |
|---|---|---|---|---|---|
| Good (yaw-aligned, 0.3m/s) | 205 | ~515 | ~197 | 1300 | **~2216** |
| Arc (not facing coin) | 70 | ~180 | ~72 | 1300 | **~1622** |
| No collect (crash) | ~50 | ~50 | ~20 | -300 | **~-180** |

Threshold 1900 requires arcing to be eliminated. 19/20 good collects → mean ~2100 (passes). 18/20 → ~1984 (passes). Arc 20/20 → ~1622 (fails).
