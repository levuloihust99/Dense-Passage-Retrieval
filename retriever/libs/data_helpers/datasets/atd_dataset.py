import json
import os
from typing import List, Dict, Text, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


def load_data(
    data_dir: Text,
    qa_file: Text,
    segmented_qa_file: Text,
    segmented: bool
) -> List[Dict[Text, Any]]:
    """Load data into a list of question, context pair."""
    logger.info("Loading AN_TOAN_DIEN question-context pairs...")
    start_time = time.perf_counter()
    if segmented:
        with open(os.path.join(data_dir, segmented_qa_file), 'r') as reader:
            qa_pairs = json.load(reader)
    else:
        with open(os.path.join(data_dir, qa_file), 'r') as reader:
            qa_pairs = json.load(reader)
    logger.info("Done loading AN_TOAN_DIEN question-context pairs in {}s".format(time.perf_counter() - start_time))
    return qa_pairs
