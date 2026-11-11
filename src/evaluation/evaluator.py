import json
import math
from typing import Any, Dict, List, Tuple

import torch


class ModelEvaluator:
    """Evaluates task accuracy, structural response validity, and token perplexity.
    Enforces Strategy #6 from Apple CHI '24: 'Create a robust evaluation pipeline before
    optimizing to reliably observe shifts in error and subpopulation behavior.'
    """

    @staticmethod
    def compute_perplexity(
        model: Any,
        tokenizer: Any,
        texts: List[str],
        max_length: int = 512,
    ) -> float:
        """Computes cross-entropy loss and exponentiates to calculate perplexity.
        Lower perplexity = better model fidelity and language modeling accuracy.
        """
        model.eval()
        device = next(model.parameters()).device
        total_loss = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for text in texts:
                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encodings.input_ids.to(device)
                if input_ids.shape[1] < 2:
                    continue

                target_ids = input_ids.clone()
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * input_ids.shape[1]

                total_loss += neg_log_likelihood.item()
                total_tokens += input_ids.shape[1]

        if total_tokens == 0:
            return float("nan")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return perplexity

    @staticmethod
    def verify_json_compliance(response: str) -> bool:
        """Verifies if the model correctly generated syntactically valid JSON."""
        clean = response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                json.loads(clean[start : end + 1])
                return True
            except json.JSONDecodeError:
                return False
        return False

    @staticmethod
    def score_task_responses(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates structural accuracy score across standard evaluation tasks."""
        scores = {}
        total = len(task_results)
        passed = 0

        for res in task_results:
            task = res.get("task", "")
            response = res.get("response", "")

            if "JSON" in task or "Instruction Following" in task:
                valid = ModelEvaluator.verify_json_compliance(response)
                scores[task] = 1.0 if valid else 0.5
            elif "Reasoning" in task:
                # Math check: The answer is 3 hours
                valid = "3" in response or "three" in response.lower()
                scores[task] = 1.0 if valid else 0.3
            elif "Code" in task:
                valid = "def " in response and "return" in response
                scores[task] = 1.0 if valid else 0.5
            else:
                valid = len(response.strip()) > 30
                scores[task] = 1.0 if valid else 0.5

            if scores[task] >= 0.8:
                passed += 1

        overall_score = (passed / total) * 100.0 if total > 0 else 0.0
        return {
            "per_task_scores": scores,
            "passed_tasks": passed,
            "total_tasks": total,
            "overall_accuracy_percentage": overall_score,
        }
