from mor_run import run_mor
from src.run.mamba_run import run_mamba
from src.run.static_recursive_mamba_run import run_static_recursive_mamba
from src.run.recursion_wise_state_run import run_recursion_wise_state
from src.run.dynamic_mor_run import run_dynamic_mor
from src.run.grouped_param_reduct_run import run_grouped_param_reduct
from src.run.lte_adaptive_width_run import run_lte_adaptive_width
from src.run.grouped_lte_run import run_grouped_lte
from src.run.turbo_state_compression_run import run_turbo_state_compression
from src.run.quantization_aware_tuning_run import run_quantization_aware_tuning
from src.run.final_joint_run import run_final_joint
from src.run.scaling_efficiency_run import run_scaling_study
from src.run.markov_head_run import run_markov_head


if __name__ == "__main__":

    run_mor()
    run_mamba()
    run_static_recursive_mamba()
    run_recursion_wise_state()
    run_dynamic_mor()

    run_grouped_param_reduct()
    run_lte_adaptive_width()
    run_grouped_lte()

    run_turbo_state_compression()
    run_quantization_aware_tuning()
    run_final_joint()

    run_scaling_study()
    run_markov_head()