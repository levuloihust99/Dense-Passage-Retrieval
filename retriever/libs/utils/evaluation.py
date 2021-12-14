import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


def calculate_metrics(eval_results, ground_truth, corpus):
    logger.info("Calculating metrics...")
    start_time = time.perf_counter()
    assert len(eval_results) == len(ground_truth)
    L = len(eval_results)
    precisions = []
    recalls = []
    f2_scores = []
    for i in range(L):
        pred_relevant_articles_ids = set(
            ["{}:::{}".format(article['law_id'], article['article_id'])
             for article in eval_results[i].get('relevant_articles')]
        )
        true_relevant_articles_ids = set(
            ["{}:::{}".format(article['law_id'], article['article_id'])
             for article in ground_truth[i].get('relevant_articles')]
        )
        pred_relevant_articles = set()
        for item in pred_relevant_articles_ids:
            law_id, article_id = item.split(':::')
            pred_relevant_articles.add(corpus[law_id][article_id]['text'])
        true_relevant_articles = set()
        for item in true_relevant_articles_ids:
            law_id, article_id = item.split(':::')
            true_relevant_articles.add(corpus[law_id][article_id]['text'])

        true_positives = 0
        for article in pred_relevant_articles:
            if article in true_relevant_articles:
                true_positives += 1

        if len(pred_relevant_articles) > 0:
            precision = true_positives / len(pred_relevant_articles)
            recall = true_positives / len(true_relevant_articles)
        else:
            precision = 0.
            recall = 0.
        if true_positives == 0:
            f2_score = 0.
        else:
            f2_score = (5 * precision * recall) / (4 * precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f2_scores.append(f2_score)
        assert eval_results[i].get(
            'question_id') == ground_truth[i].get('question_id')

    logger.info("Done calculating metrics in {}s".format(
        time.perf_counter() - start_time))

    return (
        float(np.array(precisions).mean()),
        float(np.array(recalls).mean()),
        float(np.array(f2_scores).mean())
    )
