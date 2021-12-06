import re
import multiprocessing
import logging


def remove_stopwords(text, stop_words):
    for word in stop_words:
        text = re.sub(rf'\b{word}\b', '', text)
    return text


def add_logging_info(*, shared_counter: multiprocessing.Value, logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            with shared_counter.get_lock():
                shared_counter.value += 1
                if shared_counter.value % 100 == 0:
                    logger.info("Done processing {} docs".format(shared_counter.value))
            return res
        return wrapper
    return decorator
