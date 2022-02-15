#### Table of contents
1. [Setup](#setup)
2. [Data](#data)
	- [Data specification](#data-specification)
	- [Data preparation](#data-preparation)
        - [Step 1: Word segmentation](#word-segmentation)
	    - [Step 2: Split train/test data](#split-train-test)
        - [Step 3: Generate tfrecord training data](#gen-qa-tfrecord)
3. [Training](#training)
    - [Configurations](#train-config)
    - [Run training](#run-training)
    - [Parameter descriptions](#param-description)
4. [Generate embeddings](#gen-embs)
    - [Step 1: Generate tfrecord files for corpus](#gen-corpus-tfrecord)
    - [Step 2: Generate corpus FAISS index](#faiss-indexing)
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
There are 4 files provided by Zalo AI Challenge: Legal Text Retrieval
<pre>
.
└── Legal Text retrieval
    ├── legal_corpus.json
    ├── public_test_question.json
    ├── public_test_sample_submission.json
    └── train_question_answer.json
</pre>


`train_question_answer.json` is used to train retriever model, while `legal_corpus.json` is used to evaluate the model.

## <a name="data-preparation"></a> Data prepration
### <a name="word-segmentation"></a> Step 1: Word segmentation
Word segmentation is neccessary if you use PhoBERT. We need to perform word segmentation on two files: `train_question_answer.json` and `legal_corpus.json`. The output files are `train_question_answer_segmented.json` and `legal_corpus_segmented.json`. We use `CORPUS_FILE` to refer to: (1) `legal_corpus.json` if not using PhoBERT; (2) `legal_corpus_segmented.json` if using PhoBERT. Similarly, we use `QA_FILE` to refer to: (1) `train_question_answer.json` if not using PhoBERT and (2) `train_question_answer_segmented.json` if using PhoBERT.

Sample command to run word segmentation on a `.json` file:

```shell
(.venv)$ python -m scripts.python.segment \
    --input-file data/train_question_answer.json \
    --output-file data/train_question_answer_segmented.json \
    --segment-host http://localhost:8080/segment
```
<!-- For this command to be run, you should first install `vncorenlp` and place `vncorenlp` directory at the same level of this project root. See https://github.com/VinAIResearch/PhoBERT for details about `vncorenlp` installation. -->
For this command to be run, you should first run a segmentation server somewhere and then have `--segment-host` point to that server. In this case, segmentation server runs at `http://localhost:8080` and expose an endpoint named `/segment`. A request to segmentation server must be a post request with body of `{"sentence": "The sentence you want to segment"}`.

You can follow the steps below to run a segmentation server:
```shell
$ git clone git@gitlab.com:levuloihust99/java-segmentation-server.git
$ cd java-segmentation-server
$ javac -d bin server/JsonServer.java
$ java -cp bin server.JsonServer
```
Make sure you have installed Java before running these commands.

### <a name="split-train-test"></a> Step 2: Split train/test data
Sample command to split train/test data:

```shell
(.venv)$ python -m libs.data_helpers.split \
    --train-ratio 0.8 \
    --data-path data/train_question_answer.json \
    --out-train-path data/train/train_data.json \
    --out-test-path data/test/test_data.json
```
|Param |Description |
|------|------------|
|`data_path`|Path to the `QA_FILE`|
|`out-train-path`|Data used for training, subset of data in `QA_FILE`|
|`out-test-path`|Data used for testing, subset of data in `QA_FILE`. Note that data in `out-train-path` and `out-test-path` are 2 disjoint sets.|
|`train-ratio`|Ratio of train data over all data in `QA_FILE`|

For purpose of reproducibility, two `indices.json` files are generated along with train/test data. These files hold index of train/test data in `QA_FILE`. Result of the above command should be like this:

    data
    ├── test
    │   ├── indices.json
    │   └── test_data.json
    └── train
        ├── indices.json
        └── train_data.json

### <a name="gen-qa-tfrecord"></a> Step 3: Generate tfrecord training data
`scripts/bash/sample_run_dump_qa.sh` is an example of how to generate tfrecord files used for training. Running this bashscript file is equivalent to this command:
```shell
(.venv)$ python -m scripts.python.dual_encoder.dump_qa \
    --query-max-seq-length 64 \
    --context-max-seq-length 256 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/train \
    --load-vlsp False \
    --load-zalo True \
    --load-mailong25 False \
    --load-evn-spc True \
    --load-atd True
```
You have options to use data from 5 datasets: Legal Text Retrieval (Zalo AI Challenge 2021), VLSP 2021 Machine Reading Comprehension, mailong25, EVN-SPC and ATD. You need to specify configurations for data loading in `configs/data_config.json` before running this command. An example of `configs/data_config.json` is provided in `configs/sample_data_config.json`

    {
        "vlsp": {
            "data_dir": "/home/lvloi/data/MRC/VLSP2021/training",
            "segmented": false
        },
        "zalo": {
            "data_dir": "data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/json",
            "qa_file": "train_question_answer_segmented.json",
            "corpus_file": "legal_corpus_segmented.json",
            "load_hardneg": false,
            "add_law_id": false,
            "lowercase_and_remove_indicating_words": true
        },
        "mailong25": {
            "data_dir": "/home/lvloi/data/MRC/mailong25",
            "segmented": false
        },
        "evn_spc": {
            "data_dir": "data_evn_spc/data/QA",
            "qa_file": "evn_spc_question_answer.json",
            "segmented_qa_file": "evn_spc_question_answer_segmented.json",
            "segmented": true
        },
        "atd": {
            "data_dir": "data_atd",
            "qa_file": "an_toan_dien.json",
            "segmented_qa_file": "an_toan_dien_segmented.json",
            "segmented": true
        }
    }


# <a name="training"></a> Training
## <a name="run-training"></a> Run training
`scripts/python/dual_encoder/sample_run_train.sh` is an example of running training. Running this bashscript file is equivalent to this command:
```shell
(.venv)$ python -m scripts.python.dual_encoder.train \
    --model-name phobert_ZL-EVN_B16_NonLID_Dat02 \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/train \
    --model-arch roberta \
    --sim-score dot_product \
    --query-max-seq-length 64 \
    --context-max-seq-length 256 \
    --train-batch-size 16 \
    --num-train-epochs 50 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
    --use-hardneg False \
    --use-stratified-loss False
```
If you don't want to repeatedly type such a long command to run training, you can set value of `--hparams` parameter to the path of a configuration file. The below command is equivalent the one above:
```shell
(.venv)$ python train.py --hparams train_config.json
```

    # train_config.json
    {
        "model_name": "phobert_ZL-EVN_B16_NonLID_Dat02",
        "pretrained_model_path": "pretrained/vinai/phobert-base",
        "tfrecord_dir": "data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/train",
        "model_arch": "roberta",
        "sim_score": "dot_product",
        "query_max_seq_length": 64,
        "context_max_seq_length": 256,
        "train_batch_size": 16,
        "num_train_epochs": 50,
        "logging_steps": 100,
        "save_checkpoint_freq": "epoch",
        "use_hardneg": false,
        "use_stratified_loss": false
    }
## <a name="param-description"></a> Parameter descriptions

|Param |Description |
|------|------------|
|`model_name`|A unique name for your Dual Encoder model. This name is used to create log directory and checkpoint directory for your model. Default to `dual-encoder`|
|`pretrained_model_path`|Path from which to instantiate the query encoder and context encoder, e.g. TFRobertaModel.from_pretrained($pretrained_model_path). Default to `pretrained/vinai/phobert-base`|
|`model_arch`|This parameter is used to determine which transformers class to be used to instantiate model and tokenizer. For example, if `model_arch` is `roberta`, we use `TFRobertaModel` and `PhobertTokenizer`. Allowed values for this parameters are `roberta`, `bert` and `distilbert`. See `libs/nn/constants.py` for more details.|
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
|`tfrecord_dir`|Directory that contains training data. Default to `data/tfrecord/dual_encoder/train`.|
|`checkpoint_dir`|Directory used for storing saved checkpoints. The actually checkpoint path is `$checkpoint_dir/$model_name`, e.g. `checkpoints/phophobert-base_ZL_hardneg_FrScr`. Default value is `checkpoints/dual_encoder`|
|`log_dir`|Directory used for storing logs. The actually log path is `$log_dir/$model_name`, e.g. `logs/phophobert-base_ZL_hardneg_FrScr`. Default value is `logs/dual_encoder`|
|`use_tpu`|Whether to run training on TPU. Default to `False`. If `True`, you also specify value for `tpu_name`.|
|`tpu_name`|Name of Cloud TPU. Must be specified if run training on Cloud TPU.|
|`use_hardneg`|Whether to use hard negative samples. This will change behavior of loss calculation during training. Default to `False`|
|`use_stratified_loss`|Whether to use stratified loss. If this parameter is set to `True`, `use_hardneg` must also be set to `True`|

# <a name="gen-embs"></a> Generate embeddings
## <a name="gen-corpus-tfrecord"></a> Step 1: Generate tfrecord files for corpus
`scripts/bash/sample_run_dump_corpus.sh` is an example of generating corpus tfrecords. Running this bashscript file is equivalent to this command:
Sample command:
```shell
(.venv)$ python -m scripts.python.dual_encoder.dump_corpus \
    --corpus-path data/legal_corpus_segmented.json \
    --context-max-seq-length 256 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/corpus \
    --add-law-id False
```

## <a name="faiss-indexing"></a> Step 2: Generate corpus FAISS index
`scripts/bash/sample_run_generate_embeddings.sh` is an example of indexing corpus. Running this bashscript file is equivalent to this command:

Sample command:
```shell
(.venv)$ python -m scripts.python.dual_encoder.generate_embeddings \
    --config-file logs/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/config.json \
    --index-path indexes/phobert_ZL-EVN_NonLID_Dat02 \
    --corpus-path data/legal_corpus_segmented.json \
    --corpus-tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/corpus
```

# <a name="evaluation"></a> Evaluate
After training Dual Encoder and encoding corpus, you can now evaluate your model on test data. Metrics used for evaluation are precision, recall and F2-score. 

`scripts/bash/sample_run_evaluate.sh` is an example of running evaluation. Running this bashscript file is equivalent to this command:
```shell
(.venv)$ python -m scripts.python.evaluate \
    --config-file logs/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/config.json \
    --index-path indexes/phobert_ZL-EVN_NonLID_Dat02 \
    --qa-path data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/json/test/test_data.json \
    --tokenizer-path pretrained/vinai/phobert-base \
    --result-dir results/phobert_ZL-EVN_NonLID_Dat02 \
    --batch-size 256 \
    --top-docs 100 \
    --debug \
    --write-out-results
```
Note that code for evaluation can only be run on GPU or CPU. TPU is currently not supported.
