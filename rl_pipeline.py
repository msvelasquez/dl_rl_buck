import numpy as np
from plants import EchoStateNetworkPlant

class BuckEnv:
    """Deployment-faithful Environment supporting any model from plants.py."""
    def __init__(self, plant, seed=0):
        self.plant = plant
        self.action_delay = 1
        self.ripple_coef = 0.17
        self.adc_noise_v = 0.02
        self.conditions = [(25.1, 7.2), (17.11, 7.2), (20.13, 8.7)]
        self.fixed_cond = self.conditions[0]
        
    def reset(self, seed=None):
        vin, r_load = self.fixed_cond
        v_init = vin * 0.35
        self.plant.reset(v_init=v_init, d_init=0.35)
        
        self.v_buf = [v_init] * self.plant.op_window
        self.d_buf = [0.35] * self.plant.op_window
        
        obs = np.array([v_init, 0.35, v_init / (0.35 + 1e-6)], dtype=np.float32)
        return obs, {}

    def step(self, action):
        duty = np.clip(action, 0.25, 0.85)
        v_prev = self.v_buf[-1]
        
        # Calculate trailing operating window calculations
        op_mode = np.mean(self.v_buf[-50:]) / (np.mean(self.d_buf[-50:]) + 1e-6)
        
        # Step the chosen underlying plant architecture
        inputs = np.array([[v_prev, duty, op_mode]], dtype=np.float32)
        v_next = np.clip(self.plant.forward(inputs), 0.0, 60.0)
        
        self.v_buf.append(v_next)
        self.d_buf.append(duty)
        
        obs = np.array([v_next, duty, op_mode], dtype=np.float32)
        reward = -abs(v_next - 12.0)  # tracking error penalty
        truncated = len(self.v_buf) > 400
        
        return obs, reward, False, truncated, {"v": v_next, "v_ref": 12.0}

if __name__ == "__main__":
    print("RL Pipeline engine initialized. Ready to interface stable-baselines3 algorithms.")