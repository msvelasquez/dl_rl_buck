import numpy as np

class BasePlant:
    def reset(self, v_init=0.0, d_init=0.0):
        pass
    def forward(self, inputs):
        raise NotImplementedError

class LegacyWindowPlant(BasePlant):
    """Original window-based GRU/LSTM/NARX plant."""
    def __init__(self, weights_path, meta_path):
        self.arch = "Legacy_Window"
        self.seq_len = 20  # N_OBS_SAMPLES
        self.op_window = 40
        # Load weights/meta here if needed for legacy execution
        
    def forward(self, inputs):
        # Emulates original legacy window forward pass
        return float(inputs[-1, 0] * 0.95)  # Placeholder for legacy execution

class EchoStateNetworkPlant(BasePlant):
    """Unified ESN handling v3 (Single) and v4 (Multi-Timescale) architectures."""
    def __init__(self, model_path):
        data = np.load(model_path, allow_pickle=True)
        self.model_path = model_path
        
        # Metadata parsing
        self.hs_rate = int(data.get('hs_rate', 4))
        self.steps_per_ctrl = int(data.get('steps_per_ctrl', 10))
        self.op_window = int(data.get('op_window', 50))
        self.n_groups = int(data.get('n_groups', 1))
        
        if self.n_groups == 2:
            self.arch = "v4_Multi_Timescale"
            self.n_fast = int(data['n_fast'])
            self.n_slow = int(data['n_slow'])
            self.n_res = self.n_fast + self.n_slow
        else:
            self.arch = "v3_Single_Reservoir"
            self.n_res = int(data['n_res'])
            
        # Readout matrices
        self.W_out = data['W_out']
        self.W_in = data['W_in']
        self.W_res = data['W_res']
        
        # Hyperparameters
        if self.arch == "v4_Multi_Timescale":
            self.alpha = np.concatenate([
                np.full(self.n_fast, data.get('alpha_fast', 0.40)),
                np.full(self.n_slow, data.get('alpha_slow', 0.10))
            ])
        else:
            self.alpha = float(data.get('leaking_rate', 0.10))
            
        self.vmu, self.vsd = float(data['vmu']), float(data['vsd'])
        self.omu, self.osd = float(data['omu']), float(data['osd'])
        
        self.seq_len = self.steps_per_ctrl
        self.reset()

    def reset(self, v_init=0.0, d_init=0.0):
        self.x = np.zeros(self.n_res, dtype=np.float32)
        # Warm-start initialization block if needed
        return self.x

    def forward(self, inputs):
        # expect inputs: array of shape (1, 3) -> [[v_prev, duty, op_mode]]
        v_raw, duty, op_raw = inputs[0]
        
        # Normalize inputs
        v_norm = (v_raw - self.vmu) / self.vsd
        op_norm = (op_raw - self.omu) / self.osd
        u = np.array([duty, v_norm, op_norm], dtype=np.float32)
        
        # Reservoir update
        state_update = np.tanh(np.dot(u, self.W_in) + np.dot(self.x, self.W_res))
        if isinstance(self.alpha, np.ndarray):
            self.x = (1.0 - self.alpha) * self.x + self.alpha * state_update
        else:
            self.x = (1.0 - self.alpha) * self.x + self.alpha * state_update
            
        # Linear readout
        feat = np.concatenate([u, self.x])
        v_next_norm = np.dot(feat, self.W_out)
        v_next = (v_next_norm * self.vsd) + self.vmu
        return float(v_next)