from transformers import (
    PhobertTokenizer, BertTokenizer,
    TFRobertaModel, TFBertModel
)

TOKENIZER_MAPPING = {
    'roberta': PhobertTokenizer,
    'bert': BertTokenizer
}

MODEL_MAPPING = {
    'roberta': TFRobertaModel,
    'bert': TFBertModel
}
