from dataclasses import dataclass

from src.models.quantization_aware_tuning_model import (
    QuantizationAwareTuningModelConfig,
    QuantizationAwareTuningCausalLM,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class FinalJointModelConfig(QuantizationAwareTuningModelConfig):
    pass


# ============================================================
# FINAL FLEXMAMBA
# ============================================================

class FinalJointCausalLM(QuantizationAwareTuningCausalLM):

    # ========================================================
    # LOAD S09 PARENT
    # ========================================================

    def load_qat_parent_state_dict(self, parent_state):

        missing, unexpected = self.load_state_dict(
            parent_state,
            strict=True,
        )

        if missing:
            raise RuntimeError(
                "Missing S09 parent keys:\n"
                + "\n".join(missing)
            )

        if unexpected:
            raise RuntimeError(
                "Unexpected S09 parent keys:\n"
                + "\n".join(unexpected)
            )

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        return {
            "strategy": "load_parent",
            "parent_model_type": "qat_joint_mamba",

            "transferred_parameter_count": total,
            "new_trainable_parameters": 0,

            "total_parameter_count": total,
            "transferred_parameter_fraction": 1.0,

            "quantizer_count": len(
                self.state_quantizers
            ),

            "quantizer_state_transferred": True,
        }


    # ========================================================
    # REPORTS
    # ========================================================

    def architecture_report(self):

        report = super().architecture_report()

        report.update(
            {
                "architecture": "flexmamba",
                "final_joint": True,

                "dynamic_recursive_depth": True,
                "recursion_wise_state": True,

                "grouped_parameterization": True,
                "lte_adaptive_width": True,

                "recurrent_state_quantization": True,
                "quantization_aware_training": True,

                "joint_optimization": True,

                "new_architecture_mechanism": False,
            }
        )

        return report


    def parameter_report(self):

        report = super().parameter_report()

        report.update(
            {
                "final_joint_parameters_added": 0,
            }
        )

        return report