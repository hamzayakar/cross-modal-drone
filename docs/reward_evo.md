# Architectural Manifesto: Physics & Math Justification

Before detailing the evolutionary stages of the agent's behavior, it is crucial to establish that the underlying physics engine, reward functions, and spatial geometry are mathematically sound and physically realistic. The training bottlenecks encountered were cognitive (agent capacity) rather than environmental.

## 1. Thrust-to-Weight Ratio and Hover Bias (Hierarchical PD Control)
- **Drone Mass:** $1.0 \text{ kg}$
- **Gravity:** $9.81 \text{ m/s}^2$
- **Hover Force Required:** $\approx 2.45 \text{ N}$ per motor (Total $9.81 \text{ N}$).

*(Note: The initial End-to-End architecture used a direct motor thrust mapping. With the shift to Hierarchical PD Control in Stage 0.5, this was updated to Target Attitude & Thrust.)*

The control space now maps the agent's continuous output `action[3]` (Throttle) from $[-1.0, 1.0]$ to a **Target Thrust** using the formula:
$$\text{Target Thrust} = \frac{\text{action}[3] + 1.0}{2.0} \times 20.0$$

This means a neutral output of $0.0$ yields exactly $10.0 \text{ N}$ of total thrust ($2.5 \text{ N}$ per motor), which perfectly counteracts gravity, creating a natural **Hover Bias**. The low-level PD controller then distributes this base thrust, clamped strictly at $7.5 \text{ N}$ per motor (Max $30 \text{ N}$ total), to execute the agent's target pitch, roll, and yaw commands. This provides a realistic **3:1 Thrust-to-Weight Ratio**, allowing agile recovery while strictly preventing physically impossible negative thrust.

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

## Stage 0.6: The Effort Penalty Paradox (Action Space Decoupling)

**Behavior:** After migrating to the Hierarchical PD Control architecture, the agent exhibited extreme passivity. Instead of navigating toward the targets, it preferred to output near-zero actions `[0.0, 0.0, 0.0, 0.0]`, effectively hovering in place and refusing to pitch, roll, or yaw.

**Fix:** The root cause was identified in the reward shaping function: the `effort_penalty`.
[cite_start]In the previous "End-to-End" architecture, penalizing the sum of squared actions prevented the drone from spinning its motors wildly[cite: 102]. However, in the new Hierarchical architecture, the agent's action space represents **High-Level Intent** (Target Pitch, Roll, Yaw Rate) rather than raw electrical motor effort. 

By continuing to penalize the action vector, the environment was mathematically punishing the agent for simply "making a decision to move." To eliminate this cognitive friction, the `effort_penalty_multiplier` was reduced to `0.0` in the YAML configuration. The physical motor effort is now inherently constrained and stabilized by the low-level PD controller, allowing the RL agent to freely issue attitude commands without artificial math penalties.

**Code Changes:**
```yaml
# CHANGED in configs/teacher_ppo.yaml
# OLD:
  effort_penalty_multiplier: 0.001
# NEW: Action space decoupled from raw effort; intent should not be penalized.
  effort_penalty_multiplier: 0.0

  ## Stage 0.7: The Z-Lock Trap & Curriculum Consolidation (Removing the Training Wheels)

**Behavior:** After implementing the PD controller, the agent was still being trained using "Hovercraft Mode" (`lock_z: True`), which artificially forced its Z-velocity to `0.0` and clamped its altitude to `2.0m`. Because the agent's altitude was hardcoded, any changes it made to `action[3]` (Target Thrust) had zero effect on the environment. The RL agent quickly learned that this action output was useless and stopped optimizing it. When transitioned to the next curriculum stage (`lock_z: False`), the agent catastrophically crashed because its policy had never learned to manage thrust in a 3D space.

**Fix:** Completely removed the `lock_z` constraint from the physics step. The new low-level PD controller already provides sufficient baseline stability, meaning the agent no longer needs artificial "training wheels" to survive its early steps. It can and must learn true 3D flight (managing thrust alongside attitude) from step zero. 

Because removing `lock_z` made the original Stage 0 (2D) and Stage 1 (3D) identical in their objectives, the curriculum was consolidated. The redundant stage was eliminated, streamlining the training pipeline from 6 stages down to 5 (Stage 0 to 4).

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

**Fix:** Adjusted the discount factor to mathematically match the 240Hz time domain.
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

**Fix:** Implemented three industry-standard architectural overhauls inspired by the Robotics and Perception Group (ETH Zurich):
1. **Body-Frame Coordinate Transformation:** The `relative_pos` vector (target location), `linear_vel`, and `angular_vel` were transformed from the World Frame to the Body Frame using the drone's rotation matrix. The agent now perceives targets as "forward/left/right" rather than "North/South," aligning perfectly with the future CNN's visual perspective.
2. **Rotating LiDAR (Ego-Centric Sensors):** Previously, LiDAR rays were cast in fixed global directions. The ray-casting algorithm was rewritten to multiply local ray vectors by the drone's rotation matrix. The LiDAR array now dynamically rotates with the drone's Yaw, providing interpretable and consistent obstacle detection.
3. **Aerodynamic Damping (Rotor Drag):** PyBullet's vacuum space was causing "ice-skating" overshoot behaviors. Artificial linear (`-0.5`) and angular (`-0.05`) damping forces were injected into the physics step. This synthetic air resistance allows the drone to brake naturally when attitude levels out, significantly easing the burden on the PD controller and RL policy.

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

**Behavior:** With the physics overhauled and the agent learning efficiently, a final architectural review was conducted with the ultimate project goal in mind: **Cross-Modal Policy Distillation (Teacher MLP -> Student CNN)**. The Teacher agent was performing perfectly, but its perfection was synthetic. It relied on 100% accurate absolute numerical data and was permitted to issue erratic, high-frequency command oscillations (e.g., flipping target pitch from +30 to -30 degrees instantly).

**Analysis (Scope & Sim2Real Filtering):** When preparing a Teacher for a Student CNN, the dataset generated by the Teacher must be interpretable by the Student.
1. **The Jerk Problem:** If the Teacher generates erratic actions, the CNN (which only sees a sequence of similar pixel frames) will fail to correlate visual inputs with wild label fluctuations. The Loss function will explode. The Teacher's actions must be smooth and predictable.
2. **The God-Mode Problem:** The CNN will never output perfectly precise coordinates from pixels; it will output slightly noisy estimates. If the Teacher's policy is overfitted to perfect mathematics, the system will collapse when the CNN takes over.
3. **Domain Randomization Scoping:** While real-world deployments require mass/battery randomization, this project is scoped strictly to simulation-based Cross-Modal Distillation. Therefore, physical mass randomization was explicitly discarded to maintain focus on the core problem: Vision-Based Domain Randomization (textures, lighting), which will be handled in the Student training phase.

**Fix:** - **Action Smoothness Penalty:** Replaced the discarded `effort_penalty` with a `smoothness_penalty_multiplier`. The agent is now penalized for the squared difference between the previous action ($a_{t-1}$) and the current action ($a_t$). This forces the Teacher to generate a butter-smooth flight trajectory, creating a pristine dataset for the CNN.
- **Minimal Sensor Noise:** Injected a Gaussian noise (`scale=0.01`) into the observation vector. This slight perturbation forces the Teacher policy to become robust against the inevitable micro-inaccuracies that the CNN will introduce during the Student phase.

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

**Behavior:** During architectural review for the upcoming Cross-Modal Distillation (Teacher MLP -> Student CNN), a severe "Modality Mismatch" vulnerability was identified. The Teacher was receiving Global X, Y coordinates (`drone_pos`). If the Teacher's policy optimized around global spatial memorization (e.g., "fly to absolute coordinate [3, 4]"), the Student CNN—which only perceives local pixel data—would suffer from "Perceptual Aliasing" (inability to distinguish visually identical opposite corners of the room) and fail to mimic the Teacher. Furthermore, the 16-ray LiDAR resolution was deemed too sparse, risking physical objects slipping between rays, which would poison the Teacher's logic dataset.
**Fix:** The observation space was completely purged of any "God-Mode" global positioning, while retaining mathematically perfect relative depth perception to maintain the Teacher's expert advantage.
1. **Removed Global Position:** `drone_pos[0]` (X) and `drone_pos[1]` (Y) were entirely deleted from the observation array. Only `z_altitude` was retained as it is critical for ground collision avoidance.
2. **LiDAR Resolution (The Goldilocks Zone):** Increased LiDAR rays from 16 to 36 (10-degree intervals). This closes blind spots without causing the Curse of Dimensionality. The final Observation Space expanded to a lightweight `49-D` array, which is trivially processed by the `[256, 256]` MLP while ensuring 100% ego-centric alignment with the future Student CNN.

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

**Behavior:** Although the environment physics and distillation architecture were perfected, the deep learning model (PPO) was expected to struggle with slow convergence and periodic gradient explosions. This was traced back to two fundamental neural network data-formatting traps rather than physics bugs.

**Fix 1 (The Yaw Wrap-Around Singularity):** The Yaw angle spanned from $-\pi$ to $+\pi$. A drone rotating past $180^\circ$ would experience a sudden mathematical discontinuity, jumping from $+3.14$ to $-3.14$. This $2\pi$ jump causes catastrophic gradient spikes. 
*Solution:* Decoupled the Yaw scalar into a continuous trigonometric tuple `[sin(yaw), cos(yaw)]`, mapping the rotation to a smooth unit circle. The state space increased to `50-D`.

**Fix 2 (The Curse of Unnormalized States):** The 50-D state vector contained values of vastly different scales (LiDAR: $0 \rightarrow 5$, Velocity: $-10 \rightarrow 10$, Noise: $0.01$). MLPs require inputs to follow a $\sim N(0,1)$ distribution to prevent large-magnitude inputs from dominating weight updates.
*Solution:* Wrapped the environment in Stable-Baselines3's `VecNormalize` to dynamically track running means and variances, normalizing all observations on the fly. 

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

**Behavior:** Even with a perfect observation space, the training pipeline contained three deep-rooted structural flaws spanning physics, RL mathematics, and software engineering. If left unchecked, these would have caused the model to plateau indefinitely.

**Monster 1 (Physics): Blind PD Controller.**
The low-level PD controller was receiving the drone's angular velocity (`ang_vel`) in the Global World Frame. When the drone rotated 90 degrees (Yaw), the World's Y-axis no longer aligned with the drone's Pitch axis. The PD controller attempted to stabilize Pitch but inadvertently applied torque to the Roll axis, inducing an unrecoverable death-spin.
*Fix:* The global `ang_vel` was multiplied by the transposed rotation matrix to convert it into `local_ang_vel` (Body Frame) before feeding it into the PD damping calculations.

**Monster 2 (RL Math): The Gamma Horizon Mismatch.**
To solve 240Hz myopia, the PPO agent's discount factor was set to $\gamma = 0.9995$. However, the underlying `VecNormalize` wrapper uses its own gamma to compute discounted returns for reward normalization. Its default was $0.99$. The wrapper was squashing the rewards based on a 1-second horizon, confusing the PPO agent targeting a 5-second horizon.
*Fix:* Explicitly passed `gamma=0.9995` to all `VecNormalize` instances to synchronize the time horizons.

**Monster 3 (Software Eng): Evaluation Environment Desync.**
The `eval_env` was initialized with `training=False`, meaning its normalization statistics (Mean, Variance) remained frozen at zero. When the `EvalCallback` tested the model, the model was fed incorrectly scaled raw data, causing it to fail every evaluation and never save a `best_model.zip`.
*Fix:* Engineered a custom `SyncEvalEnvCallback` that explicitly copies the running `obs_rms` from the training environment to the evaluation environment at every step.

## Stage 0.14: The Final Boss — Symmetry Breaking (Preventing Muscle-Memory Overfitting)

**Behavior:** With all systems finally synchronized, a theoretical vulnerability remained regarding the Deep Learning policy's generalization capability. The drone spawned at the exact same coordinate (`[0, 0, 2.0]`) facing exact North (`Yaw=0`) at the start of every single episode. In Reinforcement Learning, deterministic initialization allows the MLP to bypass sensor data completely and memorize a rigid "open-loop" sequence of actions (Muscle Memory) to reach the first target. When transitioned to the Cross-Modal Student CNN phase, any slight wind or initialization noise would cause catastrophic failure.

**Fix:** Implemented **Symmetry Breaking** in the environment's `reset()` function. The drone is now spawned with a randomized X/Y offset ($\pm 0.5$ meters) and a completely randomized starting Yaw angle ($-\pi$ to $+\pi$). This strict Domain Randomization forcefully prevents trajectory memorization. The agent has no choice but to actively process its Ego-Centric 50-D sensor arrays (LiDAR and Compass) from step zero to survive, guaranteeing true "closed-loop" zero-shot generalization.

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

**Fix:** Implemented **Gimbal-Stabilized LiDAR**. Scrapped the full 3x3 rotation matrix for LiDAR rays. Instead, the local rays are now rotated **only by the Yaw angle**. This perfectly mimics real-world 2D LiDAR systems: it rotates with the drone's heading but remains strictly parallel to the ground, immune to projection shrinkage caused by Pitch or Roll.

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
*Fix:* Completely rewrote the motor mixing matrix to perfectly align with the `cf2x.urdf` coordinate topology, ensuring correct sign application for all torques across all four rotors.

**Bug 2 (Reward Math): Smoothness Penalty Over-scaling.**
The jerk penalty was calculated using `np.sum(np.square(action - self.prev_action))`. Because continuous actions operate in a high-frequency (240Hz) 4-dimensional space, the raw sum produced a massive penalty per step (often overshadowing the `alive_bonus`). This paralyzed the agent, as moving any motor was mathematically worse than falling to the floor.
*Fix:* Changed the calculation to `np.mean` to normalize the penalty, and additionally softened the PD controller gains (`Kp_ang` from 8.0 to 5.0) to prevent the motors from clamping to their maximum limits during early random exploration.

**Bug 3 (Environment Geometry): The Pixel-Perfect Collection Radius.**
The $0.4 \text{ m}$ collection radius was too strict for a $36 \text{ cm}$ drone. The drone's physical body consumed $18 \text{ cm}$ of this radius, leaving a mere $22 \text{ cm}$ margin of error, making target collection a nearly impossible needle-threading task.
*Fix:* Expanded the collection radius to $0.6 \text{ m}$, providing a physically realistic "rotor wash" hit-box that rewards the agent for aggressive near-miss flybys.

**Bug 4 (RL Math): Entropy Collapse.**
PPO defaults to an entropy coefficient of $0.0$, which can lead to premature deterministic policies (e.g., spinning continuously) before the agent discovers the sparse rewards.
*Fix:* Added `ent_coef=0.01` to the PPO configuration to actively encourage early-stage exploration.

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