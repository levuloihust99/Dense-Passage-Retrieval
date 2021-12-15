import argparse
import copy
import json
import os
import time
import logging
import tensorflow as tf

from libs.nn.configuration.dual_encoder import DualEncoderConfig
from libs.nn.modeling.dual_encoder import DualEncoder
from libs.nn.optimization import get_adamw
from libs.nn.losses.dual_encoder import (
    InBatchCosineLoss, InBatchDotProductLoss, 
    StratifiedCosineLoss, StratifiedDotProductLoss
)
from libs.nn.trainer.dual_encoder import DualEncoderTrainer
from libs.nn.constants import ARCHITECTURE_MAPPINGS
from libs.utils.setup import setup_distribute_strategy, setup_memory_growth
from libs.utils.logging import add_color_formater
from libs.data_helpers.tfio.dual_encoder.loader import load_qa_dataset, load_qa_dataset_with_hardneg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


class DatasetSwitcher(object):
    def __init__(self, use_hardneg: bool = False):
        self.use_hardneg = use_hardneg
        self.load_qa_dataset_func = (
            load_qa_dataset_with_hardneg if use_hardneg
            else load_qa_dataset
        )

    def load_qa_dataset(self, *args, **kwargs):
        return self.load_qa_dataset_func(*args, **kwargs)


def main():
    # argument parser
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # dual-encoder specific
    parser.add_argument("--model-name")
    parser.add_argument("--model-arch")
    parser.add_argument("--context-max-seq-length", type=int)
    parser.add_argument("--query-max-seq-length", type=int)
    parser.add_argument("--use-hardneg", type=eval)
    parser.add_argument("--use-stratified-loss", type=eval)
    parser.add_argument("--sim-score", choices=['cosine', 'dot_product'])
    # data
    parser.add_argument("--tfrecord-dir")
    # training params
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--num-train-epochs", type=int)
    # optimization
    parser.add_argument("--lr-decay-power", type=float)
    parser.add_argument("--weight-decay-rate", type=float)
    parser.add_argument("--num-warmup-steps", type=int)
    parser.add_argument("--warmup-proportion", type=float)
    # logging
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--save-checkpoint-freq")
    parser.add_argument("--keep-checkpoint-max", type=int)
    # pretrained model path
    parser.add_argument("--pretrained-model-path")
    # tpu settings
    parser.add_argument("--use-tpu")
    parser.add_argument("--tpu-name")
    parser.add_argument("--tpu-job-name")
    parser.add_argument("--tpu-zone")
    parser.add_argument("--gcp-project")
    # model name-specific params
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--log-file")
    parser.add_argument("--config-file")
    # json file param
    parser.add_argument("--hparams", default="{}")

    args = parser.parse_args()
    if hasattr(args, 'save_checkpoint_freq'):
        save_checkpoint_freq = args.save_checkpoint_freq
        try:
            save_checkpoint_freq = int(args.save_checkpoint_freq)
        except ValueError:
            assert save_checkpoint_freq == 'epoch', \
                "`save_checkpoint_freq` should be either `epoch` or integer."
        args.save_checkpoint_freq = save_checkpoint_freq

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
    config_dir = os.path.dirname(config.config_file)
    if not tf.io.gfile.exists(config_dir):
        tf.io.gfile.makedirs(config_dir)
    with tf.io.gfile.GFile(config.config_file, 'w') as writer:
        writer.write(config.to_json_string())

    # setup environment
    setup_memory_growth()
    strategy = setup_distribute_strategy(
        use_tpu=config.use_tpu, tpu_name=config.tpu_name)

    # tensorboard setup
    if not tf.io.gfile.exists(config.tensorboard_dir):
        tf.io.gfile.makedirs(config.tensorboard_dir)
    summary_writer = tf.summary.create_file_writer(config.tensorboard_dir)

    # create dataset
    logger.info("Creating dataset...")
    start_time = time.perf_counter()
    dataset_switcher = DatasetSwitcher(use_hardneg=config.use_hardneg)
    dataset, num_examples = dataset_switcher.load_qa_dataset(
        tfrecord_dir=config.tfrecord_dir,
        query_max_seq_length=config.query_max_seq_length,
        context_max_seq_length=config.context_max_seq_length,
        train_batch_size=config.train_batch_size
    )
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )
    logger.info("Done creating dataset in {}s".format(
        time.perf_counter() - start_time))

    # instantiate model, optimizer, metrics, checkpoints within strategy scope
    with strategy.scope():
        # dual encoders
        logger.info("Instantiate dual encoder...")
        encoder_class = ARCHITECTURE_MAPPINGS[config.model_arch]['model_class']
        query_encoder = encoder_class.from_pretrained(
            config.pretrained_model_path)
        context_encoder = encoder_class.from_pretrained(
            config.pretrained_model_path)
        dual_encoder = DualEncoder(
            query_encoder=query_encoder,
            context_encoder=context_encoder
        )
        logger.info("Done instantiating dual encoder in {}s".format(
            time.perf_counter() - start_time))

        logger.info("Creating optimizer...")
        start_time = time.perf_counter()
        num_steps_per_epoch = int(
            num_examples / config.train_batch_size / strategy.num_replicas_in_sync)
        if config.num_train_epochs:
            num_train_steps = num_steps_per_epoch * config.num_train_epochs
        else:
            num_train_steps = config.num_train_steps
        num_warmup_steps = min(config.num_warmup_steps, int(
            config.warmup_proportions * num_train_steps))
        optimizer = get_adamw(
            num_train_steps=num_train_steps,
            warmup_steps=num_warmup_steps,
            learning_rate=config.learning_rate,
        )
        logger.info("Done creating optimizer in {}s".format(
            time.perf_counter() - start_time))

        logger.info("Creating loss calculator...")
        start_time = time.perf_counter()
        if config.use_hardneg and config.use_stratified_loss:
            if config.sim_score == 'cosine':
                loss_calculator = StratifiedCosineLoss()
            else:
                loss_calculator = StratifiedDotProductLoss()
        else:
            if config.sim_score == 'cosine':
                loss_calculator = InBatchCosineLoss()
            else:
                loss_calculator = InBatchDotProductLoss()

        logger.info("Creating checkpoint manager...")
        start_time = time.perf_counter()
        ckpt = tf.train.Checkpoint(
            model=dual_encoder,
            optimizer=optimizer,
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, config.checkpoint_dir, max_to_keep=5)
        logger.info("Done creating checkpoint manager in {}s".format(
            time.perf_counter() - start_time))

        # restore checkpoint or train from scratch
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            trained_steps = optimizer.iterations.numpy()
            logger.info(
                "Latest checkpoint restored -- Model trained for {} steps".format(trained_steps))
        else:
            logger.info("Checkpoint not found. Train from scratch")

    # training
    train_config = copy.deepcopy(config)
    train_config.num_train_steps = num_train_steps
    if train_config.save_checkpoint_freq == 'epoch':
        train_config.save_checkpoint_freq = num_steps_per_epoch
    trainer = DualEncoderTrainer(
        config=train_config,
        dual_encoder=dual_encoder,
        optimizer=optimizer,
        loss_calculator=loss_calculator,
        dataset=dist_dataset,
        strategy=strategy,
        ckpt_manager=ckpt_manager,
        summary_writer=summary_writer
    )
    trainer.train()


if __name__ == "__main__":
    main()
