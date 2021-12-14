import re
import multiprocessing
import logging
from typing import List, Text

logger = logging.getLogger(__name__)
shared_counter = multiprocessing.Value('i', 0) # Keep track of number of function calls.


def remove_stopwords(text, stop_words):
    for word in stop_words:
        text = re.sub(rf'\b{word}\b', '', text)
    return text


def remove_stopwords_wrapper(text: Text, stop_words: List[Text]):
    """Remove stopwords and keep track of the number of function calls

    Args:
        text (Text): The text that needs to remove stopwords from.
        stop_words (List[Text]): List of stopwords to be removed.
    """
    text = remove_stopwords(text, stop_words)
    with shared_counter.get_lock():
        shared_counter.value += 1
        if shared_counter.value % 100 == 0:
            logger.info("Done processing {} docs".format(shared_counter.value))
    return text
