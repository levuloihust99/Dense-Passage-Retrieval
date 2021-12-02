from transformers import TFDistilBertModel, TFBertModel, TFRobertaModel

ARCHITECTURE_MAPPINGS = {
    'distilbert': TFDistilBertModel,
    'bert': TFBertModel,
    'roberta': TFRobertaModel
}