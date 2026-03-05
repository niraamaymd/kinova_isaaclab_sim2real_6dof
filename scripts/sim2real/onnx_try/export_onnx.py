import torch
import torch.nn as nn
import numpy as np
import os

class KinovaNetwork(nn.Module):
    def __init__(self, input_shape=25, actions_num=6):
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        )
        self.mu_layer = nn.Linear(64, actions_num)
        
        # We define these as parameters so they are saved in the ONNX graph
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(input_shape), requires_grad=False)

    def forward(self, x):
        # Normalize input inside the model pass
        # x = (obs - mean) / sqrt(var + 1e-8)
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        x = self.actor_mlp(x)
        return self.mu_layer(x)

def export():
    checkpoint_path = os.path.abspath("../../pretrained_models/reach/final_run/nn/reach_gen3.pth")
    output_name = "kinova_gen3_policy.onnx"

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # In your printout, the weights are under the 'model' key
    sd = checkpoint['model']

    model = KinovaNetwork(input_shape=25, actions_num=6)

    # Dictionary for our clean model
    new_sd = {}
    
    # 1. Map Actor Weights
    new_sd['actor_mlp.0.weight'] = sd['a2c_network.actor_mlp.0.weight']
    new_sd['actor_mlp.0.bias'] = sd['a2c_network.actor_mlp.0.bias']
    new_sd['actor_mlp.2.weight'] = sd['a2c_network.actor_mlp.2.weight']
    new_sd['actor_mlp.2.bias'] = sd['a2c_network.actor_mlp.2.bias']
    new_sd['mu_layer.weight'] = sd['a2c_network.mu.weight']
    new_sd['mu_layer.bias'] = sd['a2c_network.mu.bias']
    
    # 2. Map Normalization Stats
    # Note: Using the exact keys from your previous printed OrderedDict
    new_sd['running_mean'] = sd['running_mean_std.running_mean']
    new_sd['running_var'] = sd['running_mean_std.running_var']

    model.load_state_dict(new_sd)
    model.eval()

    # 3. Perform Export
    dummy_input = torch.randn(1, 25)
    print(f"[INFO] Exporting to {output_name}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"✅ Success! ONNX model saved at: {os.path.abspath(output_name)}")

if __name__ == "__main__":
    export()
