# benchmarks/lm_eval_wrapper.py

from typing import List, Dict, Any

import torch


def evaluate_with_lm_eval(
    model,
    tokenizer,
    tasks: List[str],
    num_fewshot: int = 0,
    limit: int = 50,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Fuehrt lm-eval-harness Evaluation mit einem bereits geladenen Modell durch.
    Gibt das rohe Resultat von simple_evaluate zurueck.
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # HFLM akzeptiert je nach Version unterschiedliche Parameter
    try:
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
        )
    except TypeError:
        lm = HFLM(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
        )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )

    return results
