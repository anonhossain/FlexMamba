# src/evaluation/benchmark_eval.py

from transformers import AutoTokenizer
from lm_eval import evaluator

from src.evaluation.lm_eval_adapter import (
    MoREvalAdapter,
)


TASKS = [
    "lambada_openai",
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]


def get_metric(task_result, names):
    for name in names:
        for key, value in task_result.items():

            metric_name = key.split(",")[0]

            if metric_name == name:
                return float(value)

    return None


def run_benchmarks(
    model,
    cfg,
    device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["data"]["tokenizer_name"]
    )

    adapter = MoREvalAdapter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=1,
    )

    raw = evaluator.simple_evaluate(
        model=adapter,
        tasks=TASKS,

        # Let task definitions use their
        # normal evaluation configuration.
        num_fewshot=None,

        batch_size=1,
        device=str(device),
    )

    results = raw["results"]

    scores = {}

    for task in TASKS:

        task_result = results.get(
            task,
            {},
        )

        accuracy_norm = get_metric(
            task_result,
            ["acc_norm"],
        )

        accuracy = get_metric(
            task_result,
            ["acc"],
        )

        scores[task] = {
            "accuracy": accuracy,
            "accuracy_norm": accuracy_norm,
            "score": (
                accuracy_norm
                if accuracy_norm is not None
                else accuracy
            ),
        }

    # ARC in the MoR table is reported as
    # the average of Easy + Challenge.
    arc_scores = [
        scores["arc_easy"]["score"],
        scores["arc_challenge"]["score"],
    ]

    arc_scores = [
        x for x in arc_scores
        if x is not None
    ]

    scores["arc_average"] = (
        sum(arc_scores) / len(arc_scores)
        if arc_scores
        else None
    )

    # Paper-style task average
    comparison_scores = [
        scores["lambada_openai"]["score"],
        scores["hellaswag"]["score"],
        scores["piqa"]["score"],
        scores["winogrande"]["score"],
        scores["arc_average"],
        scores["mmlu"]["score"],
    ]

    comparison_scores = [
        x
        for x in comparison_scores
        if x is not None
    ]

    scores["average_accuracy"] = (
        sum(comparison_scores)
        / len(comparison_scores)
        if comparison_scores
        else None
    )

    return scores