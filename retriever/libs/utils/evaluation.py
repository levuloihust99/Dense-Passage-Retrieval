import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


def calculate_metrics(ground_truths, predictions):
    logger.info("Start evaluating...")
    start_time = time.perf_counter()

    macro_precisions = []
    macro_recalls = []
    macro_f1_scores = []
    TOP_K = len(predictions[0]["relevant_articles"])
    top_hits = [0] * TOP_K

    for truth, pred in zip(ground_truths, predictions):
        truth_ids = set([doc['article_id'] for doc in truth["relevant_articles"]])

        true_positives = 0
        best_hit_found = False
        best_hit = TOP_K
        for idx, doc in enumerate(pred["relevant_articles"]):
            if doc['article_id'] in truth_ids:
                true_positives += 1
                if not best_hit_found:
                    best_hit_found = True
                    best_hit = idx

        top_hits[best_hit:] = [v + 1 for v in top_hits[best_hit:]]

        recall = true_positives / len(truth_ids)
        if len(pred) == 0:
            precision = 0
        else:
            precision = true_positives / len(pred)
        if true_positives == 0:
            f1_score = 0
        else:
            f1_score = (2 * recall * precision) / (recall + precision)
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1_scores.append(f1_score)

    top_hits = [v / len(predictions) for v in top_hits]
    macro_precision = sum(macro_precisions) / len(macro_precisions)
    macro_recall = sum(macro_recalls) / len(macro_recalls)
    macro_f1_score = sum(macro_f1_scores) / len(macro_f1_scores)

    logger.info("Done calculating evaluation results in {}s!".format(time.perf_counter() - start_time))

    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1_score,
        'top_hits': top_hits
    }
