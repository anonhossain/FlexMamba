| YAML                                 | Changes for 16M pipeline test                      |
| ------------------------------------ | -------------------------------------------------- |
| `s00_mamba_16m.yaml`                 | `d_model: 256`, `n_layers: 8`, `max_steps: 1`      |
| `s01_mamba_30m.yaml`                 |      Do not run it                                 |
| `s02_static_recursive_mamba.yaml`    | `scale: 16m`, `d_model: 256`, `num_recursions: 3`  |
| `s03_recursion_wise_state.yaml`      | `scale: 16m`, `d_model: 256`, `num_recursions: 3`  |
| `s04_dynamic_mor.yaml`               | `scale: 16m`, `d_model: 256`, `max_recursions: 3`  |
| `s05_grouped_parameter.yaml`         | `scale: 16m`, `d_model: 256`, `max_recursions: 3`  |
| `s06_lte.yaml`                       | `scale: 16m`, `d_model: 256`, `max_recursions: 3`  |
| `s07_grouped_lte.yaml`               | `scale: 16m`, `d_model: 256`, `max_recursions: 3`  |
| `s08_turboquant.yaml`                | `scale: 16m`, `d_model: 256` if explicitly present |
| `s09_quantization_aware_tuning.yaml` | `scale: 16m`, `d_model: 256`, debug training       |
| `s10_final_joint.yaml`               | `scale: 16m`, `d_model: 256`, `max_recursions: 3`  |
| `s11_scaling_study.yaml`             | For now test only target 16M                       |
| `s12_markov_head.yaml`               | `scale: 16m`, `hidden_dim: 256`                    |
---------------------------------------------------------------------------------------------