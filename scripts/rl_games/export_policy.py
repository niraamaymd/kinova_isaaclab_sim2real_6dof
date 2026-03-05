import torch
from rl_games.torch_runner import Runner
import yaml

def export_policy():
    # 1. Load your agent config
    agent_cfg_path = "/path/to/your/pretrained_models/reach/agent.yaml"
    checkpoint_path = "/path/to/your/pretrained_models/reach/policy.pth"
    
    with open(agent_cfg_path, 'r') as f:
        agent_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # 2. Setup the Runner
    runner = Runner()
    runner.load(agent_cfg)
    
    # 3. Load the specific checkpoint weights
    # This avoids the 'constants.pkl' and 'weights_only' issues because 
    # RL-Games handles the internal dict mapping for you.
    agent = runner.create_player()
    agent.restore(checkpoint_path)
    
    # 4. Extract the model (the Actor-Critic network)
    model = agent.model
    model.eval()

    # 5. Export to TorchScript
    # Look at your env.yaml or training logs for the observation space size
    obs_shape = (1, 25) # Example: Change 48 to your actual observation dimension
    dummy_input = torch.randn(obs_shape).to(agent.device)
    
    # Trace the model
    # We trace 'model.a2c_network' or the specific policy forward pass
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("policy_compiled.pt")
    
    print("Successfully exported policy_compiled.pt")

if __name__ == "__main__":
    export_policy()
