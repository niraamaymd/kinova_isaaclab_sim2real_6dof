# test_kinova_usd.py
from omni.usd import get_context
from pxr import Usd

usd_path = "/home/robotics/KINOVA_GEN3_6DOF.usd"  # replace with actual path

stage = get_context().get_stage()
if stage is None:
    print("No stage loaded yet")
else:
    print("Stage loaded")

prim = stage.DefinePrim("/Robot", "Xform")
print(f"Trying to load USD from {usd_path}")

try:
    stage = Usd.Stage.Open(usd_path)
    print(f"USD loaded successfully: {stage.GetRootLayer().GetIdentifier()}")
except Exception as e:
    print("Failed to load USD:", e)