from enum import Enum


# pipeline names
INBATCH_PIPELINE_NAME = "inbatch"
POS_PIPELINE_NAME = "pos"
POSHARD_PIPELINE_NAME = "poshard"
HARD_PIPELINE_NAME = "hard"

# pipeline config params
# ===============================================================
# pipeline-specific
FORWARD_BATCH_SIZE = "forward_batch_size"
NUM_BACKWARD_ACCUMULATE_STEPS = "num_backward_accumulate_steps"
CONTRASTIVE_SIZE = "contrastive_size"
USE_GRADIENT_CACHE = "use_gradient_cache"
DETERMINISTIC = "deterministic"
LOGGING_STEPS = "logging_steps"

# pipeline-independent
DATA_SOURCE = "data_source"
TOKENIZER_TYPE = "tokenizer_type"
TOKENIZER_PATH = "tokenizer_path"
MAX_QUERY_LENGTH = "max_query_length"
MAX_CONTEXT_LENGTH = "max_context_length"
TRAIN_MODE = "train_mode"
LIMIT_HARDNEGS = "limit_hardnegs"
PIPELINE_SEPERATE_SYMBOL = "+"
REGULATE_FACTOR = "regulate_factor"

# inbatch-specific
USE_HARDNEG_INBATCH = "use_hardneg"
USE_NUM_HARDNEGS_INBATCH = "use_num_hardnegs"

# hard-specific
USE_HARD_NONE = "use_hard_none"

# gradient cache config
GRADIENT_CACHE_CONFIG = "gradient_cache_config"
QUERY_SUB_BATCH = "query_sub_batch"
CONTEXT_SUB_BATCH = "context_sub_batch"
# ===============================================================


class DataSourceType(object):
    ALL = "all"
    ALL_POS_ONLY = "all_pos_only"
    HARD_ONLY = "hard_only"
    HARD_NONE = "hard_none"
