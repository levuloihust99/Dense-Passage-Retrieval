import argparse
import copy
import json
import os
import time
import logging
import tensorflow as tf

from transformers import TFDistilBertModel

from dual_encoder.configuration import DualEncoderConfig
from dual_encoder.modeling import DualEncoder
from dual_encoder.optimization import get_adamw
from dual_encoder.trainer import DualEncoderTrainer
from utils.setup import setup_distribute_strategy, setup_memory_growth
from utils.logging import add_color_formater
from dataloader.loader import create_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


def main():
    # argument parser
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--model-name")
    parser.add_argument("--debug", type=eval)
    parser.add_argument("--query-max-seq-length", type=int)
    parser.add_argument("--context-max-seq-length", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--save-checkpoint-freq")
    parser.add_argument("--use-tpu", type=eval)
    parser.add_argument("--hparams", type=str, default='{}')
    args = parser.parse_args()

    args_json = copy.deepcopy(args.__dict__)
    hparams = args_json.pop('hparams')
    if args.hparams.endswith('.json'):
        with tf.io.gfile.GFile(args.hparams, "r") as f:
            hparams = json.load(f)
    else:
        hparams = json.loads(args.hparams)
    hparams = override_defaults(hparams, args_json)

    # setup logger
    add_color_formater(logging.root)

    # instantiate configuration
    config = DualEncoderConfig(**hparams)
    if not tf.io.gfile.exists(config.log_path):
        tf.io.gfile.makedirs(config.log_path)
    with tf.io.gfile.GFile(os.path.join(config.log_path, 'config.json'), 'w') as writer:
        writer.write(config.to_json_string())

    # setup environment
    setup_memory_growth()
    strategy = setup_distribute_strategy(use_tpu=config.use_tpu, tpu_name=config.tpu_name)

    # tensorboard setup
    tensorboard_dir = os.path.join(config.log_path, 'tensorboard')
    if not tf.io.gfile.exists(tensorboard_dir):
        tf.io.gfile.makedirs(tensorboard_dir)
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    
    # create dataset
    logger.info("Creating dataset...")
    start_time = time.perf_counter()
    dataset, num_examples = create_dataset(config)
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )
    logger.info("Done creating dataset in {}s".format(time.perf_counter() - start_time))

    # instantiate model, optimizer, metrics, checkpoints within strategy scope
    with strategy.scope():
        # dual encoders
        logger.info("Instantiate dual encoder...")
        query_encoder = TFDistilBertModel.from_pretrained(config.pretrained_model_path)
        context_encoder = TFDistilBertModel.from_pretrained(config.pretrained_model_path)
        dual_encoder = DualEncoder(
            query_encoder=query_encoder,
            context_encoder=context_encoder
        )
        logger.info("Done instantiating dual encoder in {}s".format(time.perf_counter() - start_time))

        logger.info("Creating optimizer...")
        start_time = time.perf_counter()
        num_steps_per_epoch = int(num_examples / config.train_batch_size / strategy.num_replicas_in_sync)
        if config.num_train_epochs:
            num_train_steps = num_steps_per_epoch * config.num_train_epochs
        else:
            num_train_steps = config.num_train_steps
        num_warmup_steps = min(config.num_warmup_steps, int(config.warmup_proportions * num_train_steps))
        optimizer = get_adamw(
            num_train_steps=num_train_steps,
            warmup_steps=num_warmup_steps,
            learning_rate=config.learning_rate,
        )
        logger.info("Done creating optimizer in {}s".format(time.perf_counter() - start_time))

        logger.info("Creating checkpoint manager...")
        start_time = time.perf_counter()
        ckpt = tf.train.Checkpoint(
            model=dual_encoder,
            optimizer=optimizer,
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, config.checkpoint_path, max_to_keep=5)
        logger.info("Done creating checkpoint manager in {}s".format(time.perf_counter() - start_time))

    # training
    train_config = copy.deepcopy(config)
    train_config.num_train_steps = num_train_steps
    if train_config.save_checkpoint_freq == 'epoch':
        train_config.save_checkpoint_freq = num_steps_per_epoch
    trainer = DualEncoderTrainer(
        config=train_config,
        dual_encoder=dual_encoder,
        optimizer=optimizer,
        dataset=dist_dataset,
        strategy=strategy,
        ckpt_manager=ckpt_manager,
        summary_writer=summary_writer
    )
    trainer.train()


if __name__ == "__main__":
    main()