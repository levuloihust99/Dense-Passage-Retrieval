import argparse
import copy
import json
import os
import time
import logging
import tensorflow as tf

from libs.nn.configuration import DualEncoderConfig
from libs.nn.modeling import DualEncoder
from libs.nn.optimization import get_adamw
from libs.nn.trainer import DualEncoderTrainer
from libs.nn.losses import LossCalculator
from libs.constants import TOKENIZER_MAPPING, MODEL_MAPPING
from libs.utils.setup import setup_distribute_strategy, setup_memory_growth, setup_random
from libs.utils.logging import add_color_formater
from libs.data_helpers.data_pipeline import get_pipelines


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


def main():
    # argument parser
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument("--random-seed", type=int)
    # dual-encoder specific
    parser.add_argument("--model-name")
    parser.add_argument("--model-arch")
    parser.add_argument("--sim-score", choices=['cosine', 'dot_product'])
    # data
    parser.add_argument("--pipeline-config-file")
    # training params
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-train-steps", type=int)
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
    parser.add_argument("--log-dir")
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

    # setup environment
    setup_memory_growth()
    setup_random(config.random_seed)
    strategy = setup_distribute_strategy(
        use_tpu=config.use_tpu, tpu_name=config.tpu_name)

    # tensorboard setup
    if not tf.io.gfile.exists(config.tensorboard_dir):
        tf.io.gfile.makedirs(config.tensorboard_dir)
    summary_writer = tf.summary.create_file_writer(config.tensorboard_dir)

    # create dataset
    logger.info("Creating dataset...")
    start_time = time.perf_counter()
    datasets = get_pipelines(config.pipeline_config)
    dist_datasets = {
        k: strategy.distribute_datasets_from_function(
            lambda _: datasets[k]
        ) for k in datasets
    }
    
    logger.info("Done creating dataset in {}s".format(
        time.perf_counter() - start_time))

    # instantiate model, optimizer, metrics, checkpoints within strategy scope
    with strategy.scope():
        # dual encoders
        logger.info("Instantiate dual encoder...")
        encoder_class = MODEL_MAPPING[config.model_arch]
        query_encoder = encoder_class.from_pretrained(
            config.pretrained_model_path, name='query_encoder')
        context_encoder = encoder_class.from_pretrained(
            config.pretrained_model_path, name='context_encoder')
        if config.model_arch == "roberta":
            query_encoder.roberta.pooler.trainable = False
            context_encoder.roberta.pooler.trainable = False
        elif config.model_arch == "bert":
            query_encoder.bert.pooler.trainable = False
            context_encoder.bert.pooler.trainable = False
        dual_encoder = DualEncoder(
            query_encoder=query_encoder,
            context_encoder=context_encoder
        )
        logger.info("Done instantiating dual encoder in {}s".format(
            time.perf_counter() - start_time))

        logger.info("Creating optimizer...")
        start_time = time.perf_counter()
        num_warmup_steps = min(config.num_warmup_steps, int(
            config.warmup_proportions * config.num_train_steps))
        optimizer = get_adamw(
            num_train_steps=config.num_train_steps,
            warmup_steps=num_warmup_steps,
            learning_rate=config.learning_rate,
        )
        logger.info("Done creating optimizer in {}s".format(
            time.perf_counter() - start_time))

        logger.info("Creating loss calculator...")
        start_time = time.perf_counter()
        loss_calculator = LossCalculator()

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
    if not tf.io.gfile.exists(config_dir):
        tf.io.gfile.makedirs(config_dir)
    with tf.io.gfile.GFile(config.config_file, 'w') as writer:
        writer.write(train_config.to_json_string())
    trainer = DualEncoderTrainer(
        config=train_config,
        dual_encoder=dual_encoder,
        optimizer=optimizer,
        loss_calculator=loss_calculator,
        datasets=dist_datasets,
        strategy=strategy,
        ckpt_manager=ckpt_manager,
        summary_writer=summary_writer
    )
    trainer.train()


if __name__ == "__main__":
    main()
