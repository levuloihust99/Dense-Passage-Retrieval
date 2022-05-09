#### Table of contents
1. [Setup](#setup)
2. [Data](#data)
	- [Data specification](#data-specification)
	- [Generate tfrecord training data](#gen-qa-tfrecord)
3. [Training](#training)
    - [Configurations](#train-config)
    - [Run training](#run-training)
    - [Parameter descriptions](#param-description)
4. [Generate embeddings](#gen-embs)
    - [Step 1: Generate tfrecord data for corpus](#gen-corpus-tfrecord)
    - [Step 2: Build FAISS index](#faiss-indexing)
5. [Evaluation](#evaluation)

# <a name="setup"></a> Setup
- Create a directory `pretrained/` to hold pretrained language models used as encoders of Dual Encoder model.
- Create virtual environment and install required packages
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements.txt
```

# <a name="data"></a> Data
## <a name="data-specification"></a> Data specification
Input data should be in JSON format. The `.json` file has the following structure:
<pre>
[
    {
        "sample_id": Integer,
        "questions": [String],
        "positive_contexts": [
            "title": String,
            "text": String
        ],
        "hardneg_contexts": [
            "title": String,
            "text": String
        ]
    }
]
</pre>

## <a name="gen-qa-tfrecord"></a> Generate tfrecord training data
There are 3 pipelines of data, i.e. `pos`, `poshard` and `hard`. The following command creates tfrecord files for `pos` pipeline:
```shell
$ python -m libs.data_helpers.build_tfrecord \
    --config-file <path to the config file> \
    --data-path <path to the JSON file> \
    --num-examples-per-file 1000 \
    --output-dir <path to the tfrecord directory> \
    --pipeline-type pos
```
The config file is in JSON format and should contains the following parameters:
- `max_query_length`
- `max_context_length`
- `tokenizer_type`
- `tokenizer_path`

E.g.
<pre>
{
    "max_query_length": 64,
    "max_context_length": 256,
    "tokenizer_type": "roberta",
    "tokenizer_path": "vinai/phobert-base"
}
</pre>

# <a name="training"></a> Training
## <a name="run-training"></a> Run training
Create a config file, e.g. `configs/training_config.json` and run the following command.
```shell
(.venv)$ python -m scripts.python.nn.train --hparams configs/training_config.json
```
The list of all possible training parameters is given in the next section. Below is an example of training config file.
<pre>
{
    "model_name": "inspect",
    "pipeline_config_file": "configs/pipeline_config.json",
    "pretrained_model_path": "vinai/phobert-base",
    "model_arch": "roberta",
    "sim_score": "dot_product",
    "num_train_steps": 10000,
    "logging_steps": 1,
    "save_checkpoint_freq": 1000,
    "learning_rate": 5e-5,
    "random_seed": 42,
    "use_tpu": true,
    "tpu_name": "node-1",
    "tpu_zone": "us-central1-a"
}
</pre>
## <a name="param-description"></a> Parameter descriptions

|Param |Description |
|------|------------|
|`model_name`|A unique name for your Dual Encoder model. This name is used to create log directory and checkpoint directory for your model. Default to `dual-encoder`|
|`pretrained_model_path`|Path from which to instantiate the query encoder and context encoder, e.g. TFRobertaModel.from_pretrained($pretrained_model_path). Default to `pretrained/vinai/phobert-base`|
|`model_arch`|This parameter is used to determine which transformers class to be used to instantiate model. For example, if `model_arch` is `roberta`, we use `TFRobertaModel`. Allowed values for this parameters are `roberta` and `bert`. See `libs/constants.py` for more details.|
|`num_train_steps`|Total number of training steps. This parameter will be overriden if `num_train_epochs` is specified.|
|`lr_decay_power`|Linear weight decay. Default to 1.0|
|`num_warmup_steps`|Number of warmup steps. Default to 1000. The actual warmup steps is `min(warmup_proportion * num_train_steps, num_warmup_steps)`.|
|`warmup_proportion`|Proportion of total steps to perform warmup. Default to 0.1|
|`logging_steps`|Print out training information (e.g. loss value) every `logging_steps` steps. Default to 100.|
|`save_checkpoint_freq`|Frequence of saving checkpoint. Default to `epoch`, meaning that checkpoint is saved every epoch. Otherwise, you can specify an integer for the value of this parameter, which means checkpoint will be save after `save_checkpoint_freq` training steps.|
|`keep_checkpoint_max`|Maximum number of checkpoints to keep. Default to 5.|
|`checkpoint_dir`|Directory used for storing saved checkpoints. The actually checkpoint path is `$checkpoint_dir/$model_name`, e.g. `checkpoints/inspect`. Default value is `checkpoints`|
|`log_dir`|Directory used for storing logs. The actually log path is `$log_dir/$model_name`, e.g. `logs/inspect`. Default value is `logs`|
|`use_tpu`|Whether to run training on TPU. Default to `False`. If `True`, you also specify value for `tpu_name`.|
|`tpu_name`|Name of Cloud TPU. Must be specified if run training on Cloud TPU.|
|`random_seed`| Set value for this parameter to allow reproducible result. If not set, the value of this param is `None`, meaning that the training process is non-reproducible.|
|`sim_score`| The similarity function used to calculate similarity between query and context, can be `cosine` or `dot_product`. Default to `dot_product`.|
|`pipeline_config_file`| Path to a JSON file contains configurations about data pipelines. Default to `configs/pipeline_config.json`.|

Data pipelines configuration is specified in another file, which corresponds to the value of `pipeline_config_file` param. Below is an example of the configuration defined in a file `configs/pipeline_config.json`.
<pre>
{
    "inbatch": {
        "use_gradient_cache": false,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "deterministic": true,
        "use_hardneg": true,
        "use_num_hardnegs": 3,
        "logging_steps": 1
    },
    "pos": {
        "use_gradient_cache": true,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "contrastive_size": 2,
        "deterministic": true,
        "logging_steps": 1
    },
    "poshard": {
        "use_gradient_cache": false,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "contrastive_size": 2,
        "deterministic": true,
        "logging_steps": 1
    },
    "hard": {
        "use_gradient_cache": false,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "contrastive_size": 2,
        "deterministic": true,
        "use_hard_none": false,
        "logging_steps": 1
    },
    "data_source": {
        "all": "data/v4/evnspc+zalo/all",
        "all_pos_only": "data/v4/evnspc+zalo/all_pos_only",
        "hard_only": "data/v4/evnspc+zalo/hard_only",
        "hard_none": "data/v4/evnspc+zalo/hard_none"
    },
    "tokenizer_type": "roberta",
    "tokenizer_path": "vinai/phobert-base",
    "max_query_length": 64,
    "max_context_length": 256,
    "train_mode": "inbatch",
    "limit_hardnegs": 50,
    "regulate_factor": 1,
    "gradient_cache_config": {
        "query_sub_batch": 2,
        "context_sub_batch": 2
    }
}
</pre>

# <a name="gen-embs"></a> Generate embeddings
## <a name="gen-corpus-tfrecord"></a> Step 1: Generate tfrecord files for corpus
The corpus should be stored in the JSON file with the following structure.
<pre>
[
    {
        "title": String,
        "text": String,
        ...
    }
]
</pre>
Assume the corpus is stored in `data/corpus.json`. The following command will create TFRecord data in the directory `data/tfrecord/corpus`.
```shell
(.venv)$ python -m libs.data_helpers.corpus_data \
    --corpus-path data/corpus.json \
    --tfrecord-dir data/tfrecord/corpus \
    --tokenizer-type roberta \
    --tokenizer-path vinai/phobert-base \
    --max-context-length 256 \
    --num-examples-per-file 5000
```
## <a name="faiss-indexing"></a> Step 2: Build corpus FAISS index
Create a `configs/indexing_config.json` file with the following content.
<pre>
{
    "use_tpu": true,
    "tpu_name": "node-1",
    "eval_batch_size": 1024,
    "max_context_length": 256,
    "checkpoint_dir": "checkpoints/inspect",
    "pretrained_model_path": "vinai/phobert-base",
    "architecture": "roberta",
    "corpus_tfrecord_dir": "data/tfrecord/corpus",
    "corpus_path": "data/corpus.json",
    "index_path": "indexes/inspect",
    "checkpoint_name": "ckpt-10"
}
</pre>
Run the following command to build FAISS index.
```shell
(.venv)$ python -m scripts.python.nn.generate_embeddings --config-file configs/indexing_config.json
```
# <a name="evaluation"></a> Evaluate
Prepare the test data in JSON format with the following structure.
<pre>
{
    "question": String,
    "relevant_articles": [
        {
            "article_id": String,
            "title": String,
            "text": String
        },
        ...
    ]
}
</pre>
Run the following command to evaluate.
```shell
(.venv)$ python -m scripts.python.nn.evaluate \
    --index-path indexes/inspect \
    --qa-path data/test_data.json \
    --result-dir results/inspect \
    --tokenizer-path vinai/phobert-base \
    --pretrained-model-path vinai/phobert-base \
    --batch-size 256 \
    --top-docs 10 \
    --max-query-length 64 \
    --checkpoint-path checkpoints/inspect/ckpt-10 \
    --architecture roberta
```