import torch
import numpy as np
import yaml
import os
import gymnasium as gym

# FIX for PyTorch 2.6+ security settings
# We use the raw strings to avoid any potential attribute errors
torch.serialization.add_safe_globals([
    'numpy.core.multiarray.scalar', 
    'numpy.core.multiarray._reconstruct', 
    'numpy.ndarray', 
    'numpy.dtype'
])

from rl_games.torch_runner import Runner
from rl_games.algos_torch import players

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self._model = model
        
    def forward(self, obs):
        normalized_obs = self._model.norm_obs(obs)
        input_dict = {'obs': normalized_obs}
        output_dict = self._model.a2c_network(input_dict)
        return output_dict['mus']

def export():
    config_path = '../../pretrained_models/reach/final_run/params/agent.yaml'
    checkpoint_path = "../../pretrained_models/reach/final_run/nn/reach_gen3.pth"
    output_name = "kinova_gen3_policy.onnx"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # UPDATED: Changed from 12 to 25 to match your checkpoint error
    num_obs = 25
    num_actions = 6

    params = config['params']
    params['config']['env_info'] = {
        'observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32),
        'action_space': gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32),
        'agents': 1
    }

    print(f"[INFO] Loading Runner with input_dim={num_obs}...")
    runner = Runner()
    runner.load(config)
    
    # Initialize player
    player = players.PpoPlayerContinuous(params)
    
    # Load weights
    print(f"[INFO] Restoring weights from {checkpoint_path}")
    # We load manually first to pass weights_only=False to the internal loader
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    player.restore(checkpoint) # Pass the loaded dict directly
    
    # Wrap and Export
    wrapper = ModelWrapper(player.model)
    wrapper.eval()

    dummy_input = torch.randn(1, num_obs)
    print(f"[INFO] Exporting to {output_name}...")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_name,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"✅ Success! Policy exported to {os.path.abspath(output_name)}")

if __name__ == "__main__":
    export()
