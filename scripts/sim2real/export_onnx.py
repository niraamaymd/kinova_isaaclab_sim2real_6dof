import torch
from rl_games.torch_runner import Runner
import yaml
import os

def export():
    # 1. Load your agent config
    with open('../../pretrained_models/reach/final_run/params/agent.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Point to the specific checkpoint you want to export
    # Ensure this matches the relative path to your .pth file
    checkpoint_path = "../../pretrained_models/reach/final_run/nn/reach_gen3.pth"
    
    config['params']['config']['checkpoint'] = checkpoint_path

    # 3. Initialize the RL-Games Runner
    runner = Runner()
    runner.load(config)

    # 4. Export to ONNX
    # RL-Games handles the mapping from the internal Actor-Critic to a pure Actor ONNX graph
    output_name = "kinova_gen3_policy.onnx"
    runner.export_onnx(output_name)
    
    print(f"✅ Success! Policy exported to {output_name}")

if __name__ == "__main__":
    export()
