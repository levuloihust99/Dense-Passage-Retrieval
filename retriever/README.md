# Setup
- Create a directory `pretrained/` to hold pretrained language models used as encoders of Dual Encoder model.
- Create virtual environment and install required packages
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements.txt
```
# Prepare data for training
## Step 1: Word segmentation

Perform word segmentation on two files: `train_question_answer.json` and `legal_corpus.json`. This is only necessary if you use PhoBERT. The output files are `train_question_answer_segmented.json` and `legal_corpus_segmented.json`. We use `CORPUS_FILE` to refer to: (1) `legal_corpus.json` if not using PhoBERT; (2) `legal_corpus_segmented.json` if using PhoBERT. Similarly, we use `QA_FILE` to refer to: (1) `train_question_answer.json` if not using PhoBERT and (2) `train_question_answer_segmented.json` if using PhoBERT.

Note that you can optionally choose to perform filtering contexts that have too large length.

## Step 2: Split train/test data
Sample command to split train/test data:

```shell
(.venv)$ python -m data_helpers.split \
    --train-ratio 0.8 \
    --data-path data \
    --output-file-name data.json \
    --qa-file train_question_answer_segmented.json
```
`data_path` and `qa_file` in combination is used to specify the actual path to `QA_FILE` (file containing all training data), which is `$data_path/$qa_file`, e.g. `data/train_question_answer_segmented.json`. the output files are `$data_path/train/$output_file_name` and `$data_path/test/$output_file_name`, e.g. `data/train/data.json` and `data/test/data.json`
## Step 3: Create question-context pairs

Create question-answer pairs based on `CORPUS_FILE` and `QA_FILE`.
- Traverse item in `QA_FILE`.
- For each item, get the question, add it to a `pair`.
- For each item, get the relevant articles.
- For each relevant article, get the `law_id` and `article_id` of it and then search for the corresponding article in `CORPUS_FILE`.
- Get the title and text of the article, add them to the `pair`.
- Add `pair` to the sequence of training data.

## Step 4: Tensorize question-context pairs
- Tokenize question and context.
- Convert tokens to integers.
- Convert those integers to `tf.Tensor`
## Step 5: Generate tfrecord files
Each example in the tfrecord file is specified as follows:

    {
        "query_input_ids": tf.TensorSpec(shape=[MAX_QUERY_LENGTH], dtype=tf.int32),
        "query_attention_mask": tf.TensorSpec(shape=[MAX_QUERY_LENGTH], dtype=tf.int32),
        "context_input_ids": tf.TensorSpec(shape=[MAX_CONTEXT_LENGTH], dtype=tf.int32),
        "context_attention_mask": tf.TensorSpec(shape=[MAX_CONTEXT_LENGTH], dtype=tf.int32)
    }

Here, `MAX_QUERY_LENGTH` and `MAX_CONTEXT_LENGTH` are the maximum tokens of a question and a context. For PhoBERT, we set `MAX_QUERY_LENGTH=64` and `MAX_CONTEXT_LENGTH=258`. See `data_helpers/dumper.py` for more details.

## Executing
Step 1 and 2 should be done before executing Steps 3, 4, 5. The following command is an example for executing Step 3, 4, 5
```shell
(.venv)$ python -m data_helpers.dumper \
    --dump-qa \
    --query-max-seq-length 64 \
    --context-max-seq-length 258 \
    --data-dir data \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --corpus-path data/legal_corpus_segmented.json \
    --qa-file data/train/data.json
```
The default locations for generated tfrecord files is `$DATA_DIR/tfrecord/train`, where `DATA_DIR` is default to `data` and can be overriden by `--data-dir` parameter.
# Training
## Configurations
Sample configurations

    data_dir = 'data'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    tokenizer_path = 'pretrained/vinai/phobert-base'
    pretrained_model_path = 'pretrained/vinai/phobert-base'

Note that `tokenizer_path` and `pretrained_model_path` must be agree with each other, i.e. the tokenizer must be exact the one used to train the pretrained model.

If you run this code on TPU, your configurations might look like belows:

    data_dir = 'gs://my-gcloud-bucket/dataset'
    checkpoint_path = 'gs://my-gcloud-bucket/checkpoints'
    log_dir = 'gs://my-gcloud-bucket/checkpoints'
    use_tpu = True
    tpu_name = 'node-1' 

## Run training
Sample training command:
```shell
(.venv)$ python train.py \
    --model-name phobertQ64C258 \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --model-arch roberta \
    --query-max-seq-length 64 \
    --context-max-seq-length 258 \
    --train-batch-size 16 \
    --num-train-steps 10000 \
    --num-train-epochs 50 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
```
The training script use `argparse` for parsing arguments. However, we do not provide options for all parameters. If you need to override parameters that do not have options provided by `argparse` or you just don't want to repeatedly type such a long command to run training, you can specify value of `--hparams` parameter to path to the configuration file. The below command is equivalent the one above:
```shell
(.venv)$ python train.py --hparams train_config.json
```

    # train_config.json
    {
        "model_name": "phobertQ64C258",
        "pretrained_model_path": "pretrained/vinai/phobert-base",
        "model_arch": "roberta",
        "query_max_seq_length": 64,
        "context_max_seq_length": 258,
        "train_batch_size": 16,
        "num_train_steps": 10000,
        "num_train_epochs": 50,
        "logging_steps": 100,
        "save_checkpoint_freq": "epoch"
    }
## Parameter descriptions

|Param |Description |
|------|------------|
|`model_name`|A unique name for your Dual Encoder model. This name is used to create log directory and checkpoint directory for your model|
|`pretrained_model_path`|Local path to a directory that contains your pretrained model or identifier of a model on HuggingFace|
|`model_arch`|This parameter is used to determine which transformers class to be used to instantiate model and tokenizer. For example, if `model_arch` is `roberta`, we use `TFRobertaModel` and `PhobertTokenizer`. Allowed values for this parameters are `roberta`, `bert` and `distilbert`. See `dual_encoder/constants.py` for more details.|
|`query_max_seq_length`|Maximum tokens of a question. Tokens that exceed `query_max_seq_length` in a question will be truncated.|
|`context_max_seq_length`|The same as `query_max_seq_length` but applying for context instead of question.|
|`train_batch_size`|Batch size used for training. In case of distributed training, this is actually batch size on each training devices. Then the global batch size is equal to the multiplication of this parameter and the number of training devices.|
|`num_train_steps`|Total number of training steps. This parameter will be overriden if `num_train_epochs` is specified.|
|`num_train_epochs`|Total number of epochs. Default to `None`. In that case, `num_train_steps` is used instead.|
|`lr_decay_power`|Linear weight decay. Default to 1.0|
|`num_warmup_steps`|Number of warmup steps. Default to 1000. The actual warmup steps is `min(warmup_proportion * num_train_steps, num_warmup_steps)`.|
|`warmup_proportion`|Proportion of total steps to perform warmup. Default to 0.1|
|`logging_steps`|Print out training information (e.g. loss value) every `logging_steps` steps. Default to 100.|
|`save_checkpoint_freq`|Frequence of saving checkpoint. Default to `epoch`, meaning that checkpoint is saved every epoch. Otherwise, you can specify an integer for the value of this parameter, which means checkpoint will be save after `save_checkpoint_freq` training steps.|
|`keep_checkpoint_max`|Maximum number of checkpoints to keep. Default to 5.|
|`data_dir`|Directory that contains training data. Default to `data`. Location of training tfrecord files is `$data_dir/tfrecord/train`. Here, `$data_dir` is the actual value of `data_dir` parameter.|
|`tokenizer_path`|Path from which to instantiate the tokenizer, e.g. `PhobertTokenizer.from_pretrained($tokenizer_path)`|
|`pretrained_model_path`|Path from which to instantiate the query encoder and context encoder, e.g. `TFRobertaModel.from_pretrained($pretrained_model_path)`.|
|`checkpoint_dir`|Directory used for storing saved checkpoints. The actually checkpoint path is `$checkpoint_dir/$model_name`, e.g. `checkpoints/phobertQ64C258`. Default value is `checkpoints`|
|`log_dir`|Directory used for storing logs. The actually log path is `$log_dir/$model_name`, e.g. `logs/phobertQ64C258`. Default value is `logs`|
|`use_tpu`|Whether to run training on TPU. Default to `False`. If `True`, you also specify value for `tpu_name`.|
|`tpu_name`|Name of Cloud TPU. Must be specified if run training on Cloud TPU.|

# Generate embeddings
## Step 1: Generate tfrecord files for corpus
Before generate embeddings, first generate tfrecord files for contexts in corpus.

Sample command:
```shell
(.venv)$ python -m data_helpers.dumper \
    --dump-corpus \
    --context-max-seq-length 258 \
    --data-dir data \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --corpus-path data/legal_corpus_segmented.json \
```
Default location for output tfrecord files is `$data_dir/tfrecord/corpus`.

## Step 2: Generate corpus embeddings
To generate embeddings, you need to load the already trained checkpoint and use context encoder from that checkpoint to encode articles from corpus. Since training configuration was saved when training Dual Encoder, you only need to specify path to the `config.json` file and no worry about other configurations. The file location is at `$log_dir/$model_name/config.json`, e.g. `logs/phobertQ64C258/config.json`. Beside, you are required to provided path to the output index files (index is performed with FAISS).

Sample command:
```shell
(.venv)$ python generate_embeddings.py \
    --config-file logs/phobertQ64C258/config.json \
    --index-path indexes/phobertQ64C258
```
Here, you do not need to specify path to the tfrecord files corresponding to the corpus. It actually does the following:

    from dual_encoder.configuration import DualEncoderConfig
    config = DualEncoderConfig.from_json_file(args.config_file)
    tfrecord_dir = os.path.join(config.data_dir, 'tfrecord', 'train)

This means that at Step 1, you should specify value of `data_dir` with the same value you specified for `data_dir` parameter when run training.

# Evaluate
After training Dual Encoder and encoding corpus, you can now evaluate your model on test data. Metrics used for evaluation are precision, recall and F2-score. Sample command for evaluating:
```shell
(.venv)$ python evaluate.py \
    --config-file logs/phobertQ64C258/config.json \
    --qa-path data/test/data.json \
    --index-path indexes/phobertQ64C258 \
    --result-dir results/phobertQ64C258 \
    --batch-size 256 \
    --top-docs 1 \
    --debug True
```
Note that code for evaluation can only be run on GPU or CPU. TPU is currently not supported.
