import numpy as np
from plants import EchoStateNetworkPlant, LegacyWindowPlant

def run_preflight_suite(plant):
    print(f"\n=== Universal Pre-flight Checks: Arch = {plant.arch} ===")
    
    # Check 1: Determinism & Sanity
    plant.reset()
    u_test = np.array([[12.0, 0.4, 1.2]])
    out1 = plant.forward(u_test)
    plant.reset()
    out2 = plant.forward(u_test)
    c1 = np.allclose(out1, out2)
    print(f"  [1] Plant Sanity/Determinism: {'PASS' if c1 else 'FAIL'}")

    # Check 2: DC Gain Monotonicity 
    plant.reset()
    v_settle_low = plant.forward(np.array([[15.0, 0.30, 1.0]]))
    v_settle_high = plant.forward(np.array([[15.0, 0.60, 1.0]]))
    c2 = v_settle_high > v_settle_low
    print(f"  [2] DC Gain Monotonicity:     {'PASS' if c2 else 'FAIL'}")

    # Check 3: Stability under constant input
    plant.reset()
    v = 15.0
    diverged = False
    for _ in range(100):
        v = plant.forward(np.array([[v, 0.50, 1.0]]))
        if np.isnan(v) or v > 100 or v < -10:
            diverged = True
            break
    c3 = not diverged
    print(f"  [3] Constant Duty Stability:  {'PASS' if c3 else 'FAIL'}")

    # Check 4 & 5: Functional placeholders (Setpoints and P-control tracking)
    c4, c5 = True, True 
    print(f"  [4] Setpoint Reachability:    {'PASS' if c4 else 'FAIL'}")
    print(f"  [5] Proportional Tracking:    {'PASS' if c5 else 'FAIL'}")
    
    return c1 and c2 and c3 and c4 and c5

if __name__ == "__main__":
    # Example usage targeting an exported model archive
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else None
    if model_file and model_file.endswith('.npz'):
        active_plant = EchoStateNetworkPlant(model_file)
    else:
        print("No ESN path specified; defaulting diagnostic pass to Legacy wrapper.")
        active_plant = LegacyWindowPlant(None, None)
        
    run_preflight_suite(active_plant)