from transformers import (
    TFDistilBertModel,
    TFBertModel,
    TFRobertaModel,
    BertTokenizer,
    PhobertTokenizer
)

ARCHITECTURE_MAPPINGS = {
    'distilbert': {
        'model_class': TFDistilBertModel,
        'tokenizer_class': BertTokenizer
    },
    'bert': {
        'model_class': TFBertModel,
        'tokenizer_class': BertTokenizer
    },
    'roberta': {
        'model_class': TFRobertaModel,
        'tokenizer_class': PhobertTokenizer
    }
}