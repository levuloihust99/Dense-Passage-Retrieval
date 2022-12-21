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
6. [Guide for NQ dataset](#nq-guide)
    - [Build index](#nq-guide-build-index)
    - [Serve index](#nq-guide-serve-index)
    - [Generate query embeddings](#nq-guide-gen-embs)
    - [Evaluation](#nq-guide-eval)
7. [Search API](#search-api)
    - [Run search API](#run-search-api)
    - [Request to search API](#request-search-api)
8. [Naming convention](#naming-convention)
    - [Data naming convention](#data-naming-convention)
    - [Checkpoint naming convention](#ckpt-naming-convention)
    - [Index naming convention](#index-naming-convention)
    - [Other naming convention](#other-naming-convention)
9. [Project orgranization](#project-org)

# <a name="setup"></a> Setup
Create virtual environment and install required packages
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements.txt
```
**Notes:** The version of Tensorflow should be compatible with cuda version installed on your machine. This repository uses Tensorflow 2.4.1, which run on cuda 11.0. If you have newer version of cuda, consider upgrading version of Tensorflow:
```shell
(.venv)$ pip install -U tensorflow tf-models-official
```

# <a name="data"></a> Data
## <a name="data-specification"></a> Data specification
Input data should be in `json` or `jsonlines` format. Each data sample has the following structure:
<pre>
{
    "sample_id": Integer,
    "questions": [String],
    "positive_contexts": [
        {
            "title": String,
            "text": String
        }
    ],
    "hardneg_contexts": [
        {
            "title": String,
            "text": String
        }
    ]
}
</pre>

## <a name="gen-qa-tfrecord"></a> Generate tfrecord training data
There are 4 types of tfrecord data, i.e. `all`, `all_pos_only`, `hard_only` and `hard_none`. The following command creates tfrecord files for `all` pipeline:
```shell
$ python -m libs.data_helpers.build_tfrecord \
    --config-file /path/to/the/config/file \
    --data-path /path/to/the/input/data \
    --data-format json \ # can be `json` or `jsonlines`
    --num-examples-per-file 1000 \
    --output-dir /path/to/the/tfrecord/directory \
    --data-type all # can be one of `all`, `all_pos_only`, `hard_only` and `hard_none`
```
The config file is in `json` format and must contain the following parameters:
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

If you want to speed up building tfrecord files for large dataset, enable multiprocessing by passing additional arguments:
```shell
$ python -m libs.data_helpers.build_tfrecord \
    --config-file /path/to/the/config/file \
    --data-path /path/to/the/input/data \
    --data-format json \ # can be `json` or `jsonlines`
    --num-examples-per-file 1000 \
    --output-dir /path/to/the/tfrecord/directory \
    --data-type all \ # can be one of `all`, `all_pos_only`, `hard_only` and `hard_none`
    --parallelize True \
    --num-processes 10 # default to the number of CPUs
```

# <a name="training"></a> Training
## <a name="run-training"></a> Run training
Create a config file, e.g. `configs/training_config.json` and run the following command.
```shell
(.venv)$ python -m scripts.python.nn.train --hparams configs/training_config.json
```
The list of all possible training parameters is given in the next section. Below is an example of training config file.
<pre>
{
    "model_name": "Luffy",
    "pipeline_config_file": "configs/training_configs/zalo2021/segmented/Luffy/pipeline_config.json",
    "pretrained_model_path": "vinai/phobert-base",
    "model_arch": "roberta",
    "sim_score": "dot_product",
    "num_train_steps": 10000,
    "logging_steps": 10,
    "num_warmup_steps": 1000,
    "save_checkpoint_freq": 1,
    "learning_rate": 5e-5,
    "random_seed": 42,
    "checkpoint_dir": "checkpoints/zalo2021/segmented",
    "log_dir": "logs/zalo2021/segmented"
}
</pre>
## <a name="param-description"></a> Parameter descriptions

|Param |Description |
|------|------------|
|`debug`|If `True`, run 1 forward and backward pass for all types of computations to check if different computations are equivalent. Default to `False`.|
|`model_name`|A unique name for your Dual Encoder model. This name is used to create log directory and checkpoint directory for your model. Default to `dual-encoder`.|
|`model_arch`|This parameter is used to determine which transformers class to be used to instantiate model. For example, if `model_arch` is `roberta`, we use `TFRobertaModel`. Allowed values for this parameters are `roberta` and `bert`. See `libs/constants.py` for more details. Default to `roberta`.|
|`sim_score`| The similarity function used to calculate similarity between query and context, can be `cosine` or `dot_product`. Default to `dot_product`.|
|`pipeline_config_file`| Path to a json file contains configurations about data pipelines. Default to `configs/pipeline_config.json`.|
|`learning_rate`|Learning rate for the AdamW optimizer. Default to 5e-5.|
|`num_train_steps`|Total number of training steps. You should calculate this number based on the dataset size and the number of equivalent training epochs that you want to perform. Default to 10000.|
|`lr_decay_power`|Linear weight decay. Default to 1.0.|
|`weight_decay_rate`|Weight decay rate for AdamW optimizer. Default to 0.01.|
|`num_warmup_steps`|Number of warmup steps. Default to 1000. The actual warmup steps is `min(warmup_proportion * num_train_steps, num_warmup_steps)`.|
|`warmup_proportion`|Proportion of total steps to perform warmup. Default to 0.1.|
|`max_grad_norm`|Max gradient norm used for gradient clipping. Default to 1.0.|
|`adam_eps`|Epsilon parameter of AdamW optimizer. Default to 1e-6.|
|`beta_1`|Beta 1 parameter of AdamW optimizer. Default to 0.9.|
|`beta_2`|Beta 2 parameter of AdamW optimizer. Default to 0.999.|
|`logging_steps`|Print out training information (e.g. loss value) every `logging_steps` steps. Default to 100.|
|`save_checkpoint_freq`|Frequency of saving checkpoint. Default to `epoch`, meaning that checkpoint is saved every epoch. Otherwise, you can specify an integer for the value of this parameter, which means checkpoint will be save after `save_checkpoint_freq` training steps.|
|`keep_checkpoint_max`|Maximum number of checkpoints to keep. Default to 5.|
|`pretrained_model_path`|Path from which to instantiate the query encoder and context encoder. Default to `pretrained/vinai/phobert-base`.|
|`use_tpu`|Whether to use TPU for training. Default to `False`.|
|`tpu_name`|`tpu_name` parameter to connect to TPU cluster. Default to `None`.|
|`tpu_job_name`|`tpu_job_name` parameter to connect to TPU cluster. Default to `None`.|
|`tpu_zone`|`tpu_zone` parameter to connect to TPU cluster. Default to `None`.|
|`gcp_project`|`gcp_project` parameter to connect to TPU cluster. Default to `None`.|
|`checkpoint_dir`|Directory used for storing checkpoints. The actually checkpoint path is `[checkpoint_dir]/[model_name]`. Default value is `checkpoints/[model_name]`.|
|`log_dir`|Directory used for storing logs. The actually log path is `[log_dir]/[model_name]`. Default value is `logs/[model_name]`.|

Data pipelines configuration is specified in another file, which corresponds to the value of `pipeline_config_file` param. Below is an example of the configuration defined in a file `configs/pipeline_config.json`.
<pre>
{
    "inbatch": {
        "use_gradient_cache": true,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "use_gradient_accumulate": true,
        "use_hardneg": true,
        "use_num_hardnegs": 1,
        "deterministic": false,
        "logging_steps": 10,
        "shuffle_positive": true
    },
    "pos": {
        "use_gradient_cache": true,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "use_gradient_accumulate": true,
        "contrastive_size": 7,
        "deterministic": false,
        "logging_steps": 10,
        "shuffle_positive": true
    },
    "poshard": {
        "use_gradient_cache": true,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "use_gradient_accumulate": true,
        "contrastive_size": 7,
        "deterministic": false,
        "logging_steps": 10,
        "shuffle_positive": true
    },
    "hard": {
        "use_gradient_cache": true,
        "forward_batch_size": 2,
        "num_backward_accumulate_steps": 1,
        "use_gradient_accumulate": true,
        "contrastive_size": 7,
        "use_hard_none": true,
        "deterministic": false,
        "logging_steps": 10
    },
    "data_source": {
        "all": "data/zalo2021/train/tfrecord/segmented/all",
        "all_pos_only": "data/zalo2021/train/tfrecord/segmented/all_pos_only",
        "hard_only": "data/zalo2021/train/tfrecord/segmented/hard_only",
        "hard_none": "data/zalo2021/train/tfrecord/segmented/hard_none"
    },
    "max_query_length": 64,
    "max_context_length": 256,
    "tokenizer_type": "roberta",
    "tokenizer_path": "vinai/phobert-base",
    "train_mode": "inbatch+pos+poshard+hard",
    "regulate_factor": 1,
    "limit_hardnegs": -1,
    "shuffle_buffer_size": 5000,
    "gradient_cache_config": {
        "query_sub_batch": 16,
        "context_sub_batch": 16
    }
}
</pre>
**Notes**: Never set `use_gradient_cache=true` and `use_gradient_accumulate=false` at the same time. It causes Gradient-Cache to behave wrong.
# <a name="gen-embs"></a> Generate embeddings
## <a name="gen-corpus-tfrecord"></a> Step 1: Generate tfrecord files for corpus
The corpus should be stored in the `json` or `jsonlines` format. Each item/passage of the corpus should has the following structure:
<pre>
{
    "article_id": String | Integer,
    "title": String,
    "text": String,
    ...
}
</pre>
The field `article_id` is neccessary for evaluation, it allows to perform matching between two passages. You can add arbitrarily other fields as metadata to enrich the search results.

Assume the corpus is stored in `data/corpus.json`. The following command will create TFRecord data in the directory `data/tfrecord/corpus`.
```shell
(.venv)$ python -m libs.data_helpers.corpus_data \
    --corpus-path data/zalo2021/corpus/json/corpus_segmented.json \
    --tfrecord-dir data/zalo2021/corpus/tfrecord/segmented \
    --tokenizer-type roberta \
    --tokenizer-path vinai/phobert-base \
    --max-context-length 256 \
    --num-examples-per-file 5000 \
    --num-processes \ # if greater than 1, use multiprocessing
    --data-format json \ # can be `json` or `jsonlines`
```
## <a name="faiss-indexing"></a> Step 2: Build corpus FAISS index
Create a `configs/indexing_config.json` file with the following content.
<pre>
{
    "use_tpu": false,
    "tpu_name": "",
    "eval_batch_size": 256,
    "max_context_length": 256,
    "checkpoint_dir": "checkpoints/zalo2021/segmented/Luffy",
    "pretrained_model_path": "vinai/phobert-base",
    "architecture": "roberta",
    "corpus_tfrecord_dir": "data/zalo2021/corpus/tfrecord/segmented",
    "corpus_path": "data/zalo2021/corpus/json/corpus_meta.json",
    "corpus_format": "json",
    "index_path": "indexes/corpus[zalo2021]/model[zalo2021_segmented_Luffy_ckpt-1]",
    "checkpoint_name": "ckpt-1",
    "num_embeddings_per_file": 100000,
    "build_index": true,
    "embedding_dir": "embeddings/corpus[zalo2021]/model[zalo2021_segmented_Luffy_ckpt-1]",
    "corpus_size": 61425,
    "skip_size": 0,
    "skip_counter": 0
}
</pre>
Run the following command to build FAISS index or generate embeddings only.
```shell
(.venv)$ python -m scripts.python.nn.generate_embeddings --config-file configs/indexing_config.json
```
### <a name="gen-embs-param-desc"></a> Parameters description
|Param |Description |
|------|------------|
|`use_tpu`|Whether to use TPU for generating embeddings. Kaggle offers free TPU v3-8 so you can use TPU on Kaggle by setting this parameter to `True`. This parameter is required.|
|`tpu_name`|If you use Kaggle or Google Colab, this should be empty string. If you use Virtual Machine on Google Cloud, this should be the name of the TPU node.|
|`eval_batch_size`|Batch size for generating embeddings. This value is per one device. The actual batch size should be multiplied by the number of devices used for generating embeddings.|
|`max_context_length`|This parameter is used when loading corpus dataset from tfrecord files. It should exactly be equal to the `max_context_length` parameter set when generating the tfrecord files.|
|`checkpoint_dir`|Path to the directory containing checkpoint files. The path can be a Google Cloud Storage path, i.e. starts with `gs://`|
|`pretrained_model_path`|Path to the pretrained model, this is neccessary to instantiate weights for model since Tensorflow uses lazy weight initialization.|
|`architecture`|Can be `roberta` or `bert`, used to determine which transformers class used to instantiate the dual encoder.|
|`corpus_tfrecord_dir`|Path to the directory containing tfrecord files of the corpus. Guide for creating these files is in [Generate tfrecord files for corpus](#gen-corpus-tfrecord)|
|`corpus_path`|Path to the corpus file. This is not the actual corpus that was serialized into tfrecord format but should be the proper data that will be returned as search results. For example, the proper data is `{"title": "Who is Einstein?", "text": ...}` but the actual data contained in tfrecord file is `{"title": "who is einstein?", "text": ...}`, i.e. tfrecord files contained data that was transformed by lowercasing, removing stopwords, etc.|
|`corpus_format`|Can be `json` or `jsonlines`. If `json`, the corpus file (specified by `corpus_path` param) should be a list of passages. If `jsonlines`, each line in the corpus file is a passage.|
|`build_index`|Take boolean value. If `True`, create FAISS index. Otherwise, only generate the embedding vectors for the corpus and save them in `.pkl` files. These `.pkl` files contains passage embeddings along with their `article_id`.|
|`index_path`|Path to the FAISS index directory. If not exists, an empty directory will be created. If `build_index=false`, this param is ignored.|
|`embedding_dir`|Path to the directory containing `.pkl` files that contains embedding vectors of the corpus. If `build_index=true`, this param is ignored.|
|`num_embeddings_per_file`|Number of embedding vectors contained in one `.pkl` file.|
|`corpus_size`|Size of the corpus. The size of the corpus should be known in advance, allowing the use of TQDM progress bar.|
|`skip_size`|An integer value denoting how many passages have already been processed. It is used to resume the generating embeddings process.|
|`skip_counter`|An integer value denoting how many `.pkl` files have already been generated. It is used to resume the generating embeddings process.|

If you set `build_index=true`, a further step needs to be done for building FAISS index
```shell
(.venv)$ python -m scripts.python.nn.build_index --embedding-dir embeddings/data[cmc]_model[zalo2021] --index-path indexes/data[cmc]_model[zalo2021]
```

# <a name="evaluation"></a> Evaluate
Prepare the test data in `json` format with the following structure.
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
    --index-path indexes/corpus[zalo2021]/model[zalo2021_segmented_Luffy_ckpt-1] \
    --qa-path data/zalo2021/test/json/test_segmented.json \
    --result-dir results/corpus[zalo2021]/model[zalo2021_segmented_Luffy_ckpt-1] \
    --tokenizer-path vinai/phobert-base \
    --pretrained-model-path vinai/phobert-base \
    --batch-size 256 \
    --top-docs 10 \
    --max-query-length 64 \
    --checkpoint-path sample/checkpoints/zalo2021/segmented/Luffy/ckpt-1 \
    --architecture roberta
```
## <a name="eval-param-desc"></a> Parameters description
We only note here some important parameters
|Param |Description |
|------|------------|
|`index_path`|Path to the FAISS index directory.|
|`qa_path`|Path to the test data file.|
|`result_dir`|Path to the directory that will contain the evaluation results.|
|`checkpoint_path`|Path to the checkpoint prefix, specified by Tensorflow. E.g. `ckpt-10`.|

# <a name="nq-guide"></a> Guide for NQ dataset
The NQ dataset does not have the `article_id` field for test data. It performs matching by checking if the answers are contained in the retrieval results. Thus, we write a separate code for evaluation of NQ dataset.

## <a name="nq-guide-build-index"></a> Build index
The FAISS index of NQ dataset (21,015,324 passages) takes about 60GB RAM when running and even more RAM when loading from disk. It usually requires a machine with 90GB RAM for loading FAISS index of NQ dataset. In case we do not have such a machine, we break the FAISS index into multiple parts, each is loaded by one machine. To do this, run the following command:
```shell
(.venv)$ python -m scripts.python.nn.build_index \
    -embedding-dir embeddings/corpus[zalo2021]/model[zalo2021_segmented_Luffy_ckpt-1] \
    --index-path indexes/corpus[zalo2021]/model[zalo2021_segmented_Luffy_ckpt-1] \
    --start-embedding-file 10 \
    --num-embedding-files 10
```
Running this command creates a FAISS index with only a portion of the corpus, i.e. only passages contained in `.pkl` files from the 10-th file to the 20-th file are included.

## <a name="nq-guide-serve-index"></a> Serve index
To serve the index, run the following command:
```shell
(.venv)$ python -m scripts.python.index_serve --index-path /path/to/the/index/directory --port 5055
```
Serving the index is required for running later evaluation script on NQ dataset.

## <a name="nq-guide-gen-embs"></a> Generate query embeddings
To avoid repeatedly generating embeddings for questions in the test dataset, we can save the generated embeddings to disk. Run the following command to generate question embeddings once and save them into disk.
```shell
(.venv)$ python -m libs.utils.nq_utils.generate_query_embeddings \
    --checkpoint-path /path/to/the/checkpoint \
    --architecture roberta \ # `bert` or `roberta`
    --pretrained-model-path vinai/phobert-base \
    --max-query-length 64 \
    --batch-size 16 \
    --qas-path /path/to/test/data \ # in `json` format
    --embedding-path # path to save the generated query embeddings
```

## <a name="nq-guide-eval"></a> Evaluation
```shell
(.venv)$ python -m libs.utils.nq_utils.nq_validate \
    --cache-path /path/to/the/query/embeddings \
    --index-hosts http://[IP1:port1],http://[IP2:port2] \ # addresses of the machine serving the index, separated by comma
    --num-queries-per-requests 50 \ # number of queries used in one search request to a machine serving the index
    --top-docs \ # number of top passages to be returned
    --corpus-size 21015324 \ # size of the corpus
    --corpus-path  # path to the corpus file, `jsonlines` format
    --qas-path  # path to the test data, used to get the answers for each question
    --result-dir # path to the result directory
```

# <a name="search-api"></a> Search API
## <a name="run-search-api"></a> Run search API

```shell
(.venv)$ python -m scripts.python.api_serve \
    --architecture roberta \
    --tokenizer-path vinai/phobert-base \
    --pretrained-model-path vinai/phobert-base \
    --checkpoint-path /path/to/checkpoint/file \
    --index-path /path/to/index/path \
    --port 5053
```

## <a name="request-search-api"></a> Request to search API
<pre>
POST http://[IP:5053]/retrieve

{
    "query": String,
    "top_docs": Integer
}
</pre>

# <a name="naming-convention"></a> Naming convention
For efficient management of checkpoints, data and indexes, we use the following naming convention.

## <a name="data-naming-convention"></a> Data naming convention
Data naming convention is based following statements:

- Each dataset has a name. We can create various versions from the original dataset, e.g. lowercase, segment, remove stopwords, etc.

- Each dataset is originally stored in `json` or `jsonlines` format. Whereas, training data is stored in `tfrecord` format.

- Each dataset has 2 parts: corpus and training data.

Below is an example of data directory organization.
<pre>
data
└── zalo2021
    ├── train
    │   ├── json
    │   │   └── train_segmented.json
    │   └── tfrecord
    │       └── segmented
    │           ├── all/
    │           ├── all_pos_only/
    │           ├── hard_only/
    │           └── hard_none/
    └── corpus
        ├── json
        │   └── corpus.json
        └── tfrecord/
</pre>
In this tree structure, `zalo2021` is the dataset name, `segmented` is the data version

## <a name="ckpt-naming-convention"></a> Checkpoint naming convention

- Checkpoint should be saved at the path of `checkpoints/[datasetname]/[datasetversion]/[model_name]/`.

    For example,
    ```
    checkpoints/zalo2021/segmented/Luffy/
    ```

- By this convention, we easily know what dataset and dataset version the checkpoint was trained on. Further parameters when training model can be found by the model name, i.e. training configuration is stored at `logs/zalo2021/segmented/Luffy/config.json`.

## <a name="index-naming-convention"></a> Index naming convention
- Index should be saved at the path of `indexes/corpus[[corpus]]/model[[traindataset]_[datasetversion]_[model_name]_[checkpoint_name]]`.

    For example,
    ```
    indexes/corpus[cmc]/model[zalo2021_segmented_Luffy_ckpt-10]
    ```
- By this convention, we easily know which corpus this index contains and which model was used to generate embedding vectors for passages in the corpus.

## <a name="other-naming-convention"></a> Other naming conventions

### Other naming conventions includes:

- Logging directory
- Embedding directory
- Cached query embedding file
- Result directory
- Training config directory
- Indexing config directory

### Conventions:

- **Logging directory**: `logs/[dataset]/[datasetversion]/`.

    E.g. 
    ```
    logs/zalo2021/segmented
    ```
- **Embedding directory**: `embeddings/corpus[[corpus]]/model[[traindataset]_[datasetversion]_[model_name]_[checkpoint_name]]`.

    E.g. 
    ```
    embeddings/corpus[cmc]/model[zalo2021_segmented_Luffy_ckpt-10]
    ```

- **Cached query embedding file**: `cached_query_embeddings/testdataset[[testdataset]]/model[[traindataset]_[datasetversion]_[model_name]_[checkpoint_name]]/query_embeddings.pkl`.

    E.g.
    ```
    cached_query_embeddings/testdataset[evnspc]/model[zalo2021_segmented_Luffy_ckpt-10]/query_embeddings.pkl
    ```
- **Result directory**: `results/corpus[[corpus]]/model[[traindataset]_[datasetversion]_[model_name]_[checkpoint_name]]`

    E.g.
    ```
    results/corpus[cmc]/model[zalo2021_segmented_Luffy_ckpt-10]
    ```

- **Training config directory**: `training_configs/[traindataset]/[datasetversion]/[model_name]`

    E.g.
    ```
    training_configs/zalo2021/segmented/Luffy
    ```

- **Indexing config directory**: `indexing_configs/corpus[[corpus]]/model[[traindataset]_[datasetversion]_[model_name]_[checkpoint_name]]`

    E.g.
    ```
    indexing_configs/corpus[cmc]/model[zalo2021_segmented_Luffy_ckpt-10]
    ```

# <a name="project-org"></a> Project organization
Below is an example of the directory structure of this project:
<pre>
.
├── cached_query_embeddings
│   └── testdataset[zalo2021]
│       └── model[zalo2021_segmented_Luffy_ckpt-1]
│           └── query_embeddings.pkl
├── checkpoints
│   └── zalo2021
│       └── segmented
│           └── Luffy
│               ├── checkpoint
│               ├── ckpt-1.data-00000-of-00001
│               └── ckpt-1.index
├── configs
│   ├── indexing_configs
│   │   └── corpus[zalo2021]
│   │       └── model[zalo2021_segmented_Luffy_ckpt-1]
│   │           └── indexing_config.json
│   └── training_configs
│       └── zalo2021
│           └── segmented
│               └── Luffy
│                   ├── pipeline_config.json
│                   └── training_config.json
├── data
│   └── zalo2021
│       ├── corpus
│       │   ├── json
│       │   │   ├── corpus_meta.json
│       │   │   └── corpus_segmented.json
│       │   └── tfrecord
│       │       └── segmented
│       │           ├── corpus_000.tfrecord
│       │           ├── corpus_001.tfrecord
│       │           ├── corpus_002.tfrecord
│       │           ├── corpus_003.tfrecord
│       │           ├── corpus_004.tfrecord
│       │           ├── corpus_005.tfrecord
│       │           ├── corpus_006.tfrecord
│       │           ├── corpus_007.tfrecord
│       │           ├── corpus_008.tfrecord
│       │           ├── corpus_009.tfrecord
│       │           ├── corpus_010.tfrecord
│       │           ├── corpus_011.tfrecord
│       │           └── corpus_012.tfrecord
│       ├── test
│       │   └── json
│       │       ├── test_segmented.json
│       │       └── test_segmented.jsonl
│       └── train
│           ├── json
│           │   └── train_segmented.json
│           └── tfrecord
│               └── segmented
│                   ├── all
│                   │   ├── data_000.tfrecord
│                   │   ├── data_001.tfrecord
│                   │   ├── data_002.tfrecord
│                   │   ├── data_003.tfrecord
│                   │   └── data_004.tfrecord
│                   ├── all_pos_only
│                   │   ├── data_000.tfrecord
│                   │   ├── data_001.tfrecord
│                   │   ├── data_002.tfrecord
│                   │   ├── data_003.tfrecord
│                   │   └── data_004.tfrecord
│                   ├── hard_none
│                   │   └── data_000.tfrecord
│                   └── hard_only
│                       ├── data_000.tfrecord
│                       ├── data_001.tfrecord
│                       ├── data_002.tfrecord
│                       └── data_003.tfrecord
├── embeddings
│   └── corpus[zalo2021]
│       └── model[zalo2021_segmented_Luffy_ckpt-1]
│           └── corpus_embedding_splitted_000.pkl
├── indexes
│   └── corpus[zalo2021]
│       └── model[zalo2021_segmented_Luffy_ckpt-1]
│           ├── index.dpr
│           └── index_meta.dpr
├── logs
│   └── zalo2021
│       └── segmented
│           └── Luffy
│               ├── config.json
│               └── tensorboard
│                   └── events.out.tfevents.1671588637.aivm01.2483670.0.v2
└── results
    └── corpus[zalo2021]
        └── model[zalo2021_segmented_Luffy_ckpt-1]
            └── metrics.txt
</pre>