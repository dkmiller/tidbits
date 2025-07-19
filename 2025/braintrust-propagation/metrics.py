def precision_recall_score(input: str, output: list[str], expected: list[str]):
    true_positives = [item for item in output if item in expected]
    false_positives = [item for item in output if item not in expected]
    false_negatives = [item for item in expected if item not in output]
 
    precision = len(true_positives) / (len(output) or 1)
    recall = len(true_positives) / (len(expected) or 1)
 
    return {
        "name": "PrecisionRecallScore",
        "score": (precision + recall) / 2,  # F1-style simple average
        "metadata": {
            "truePositives": true_positives,
            "falsePositives": false_positives,
            "falseNegatives": false_negatives,
            "precision": precision,
            "recall": recall,
        },
    }
