# Setup
- Create a directory `pretrained/` to hold pretrained language models used as encoders of Dual Encoder model.
- Create virtual environment and install required packages
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements.txt
```
# Training
## Configurations
The following parameters are recommended when run training on GPU

    data_dir: dataset
    checkpoint_path: checkpoints
    log_dir: logs
    tokenizer_path: pretrained/[name-of-the-pretrained-model]
    pretrained_model_path: pretrained/[name-of-the-pretrained-model]
If you run this code on TPU, you should change the `data_dir`, `checkpoint_path`, `log_dir` to path on google cloud storage. For examples:

    data_dir: gs://[bucket-name]/dataset
    checkpoint_path: gs://[bucket-name]/checkpoints
    log_dir: gs://[bucket-name]/checkpoints

Additionally, specify several following TPU settings:

    use_tpu: True
    tpu_name: [your-tpu-node-name]

## Run training
Run the following command to train Dual Encoder model:
```shell
(.venv)$ python train.py \
    --model-name [your-model-name] \
    --query-max-seq-length 64 \
    --context-max-seq-length 512 \
    --num-train-epochs 50 \
    --train-batch-size 256 \
    --tokenizer-path pretrained/[your-pretrained-model] \
    --pretrained-model-path pretrained/[your-pretrained-model]
```
# Generate embeddings
To generate embeddings, you need to load the already trained checkpoint and use context encoder from that checkpoint to encode articles from corpus. Since training configuration was saved when training Dual Encoder, you only need to specify path to the `config.json` file and no worry about other configurations. The file location is at `[log_dir]/[model_name]/config.json`. Beside, you are required to provided path to the output index files (index is performed with FAISS).

```shell
(.venv)$ python generate_embeddings.py \
    --config-file [log_dir]/[model_name]/config.json \
    --index-path [index_path]
```
# Evaluate
After training Dual Encoder and encoding corpus, you can now evaluate your model on test data. Metrics used for evaluation are precision, recall and F2-score. Run the following command to perform evaluating
```shell
(.venv)$ python evaluate.py \
    --config-file [log_dir]/[model_name]/config.json \
    --corpus-path [data_dir]/legal_corpus.json \
    --index-path [index_path] \
    --result-dir [res_dir] \
    --batch-size 256 \
    --top-docs 1 \
    --debug True
```
Note that code for evaluation can only be run on GPU or CPU. TPU is currently not supported.
