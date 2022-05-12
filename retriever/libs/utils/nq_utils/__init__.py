import re
import logging
import string
import collections

from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict


logger = logging.getLogger(__name__)
QAMatchStats = collections.namedtuple("QAMatchStats", ["top_k_hits", "questions_doc_hit"])