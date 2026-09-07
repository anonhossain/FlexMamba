import yaml
from models.mor_small import MoRConfig, MoRCausalLM

with open("configs/mor_16m.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = MoRCausalLM(MoRConfig(**cfg["model"]))
report = model.parameter_report()
for k, v in report.items():
    print(f"{k}: {v:,} ({v/1e6:.3f}M)")

print(f"Maximum effective block depth: {1 + cfg['model']['n_shared_layers'] * cfg['model']['num_recursions'] + 1}")
