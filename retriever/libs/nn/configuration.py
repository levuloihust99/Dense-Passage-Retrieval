import os
import json
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class CommonConfig(object):
    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v

    def to_json_string(self):
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_obj):
        return cls(**json_obj)

    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, 'r') as reader:
            json_obj = json.load(reader)
        return cls.from_json(json_obj)


class DualEncoderConfig(CommonConfig):
    def __init__(self, **kwargs):
        # dual-encoder-specific
        self.model_name = 'dual-encoder'
        self.model_arch = 'roberta'
        self.sim_score = 'dot_product'

        # data pipeline
        self.pipeline_config_file = 'configs/pipeline_training_config.json'
        self.regulate_factor = None

        # training configurations
        self.learning_rate = 5e-5
        self.eval_batch_size = 256
        self.num_train_steps = 10000

        # optimization
        self.lr_decay_power = 1.0  # Linear weight decay by default
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 1000
        self.warmup_proportions = 0.1

        # logging
        self.logging_steps = 100
        self.save_checkpoint_freq = self.logging_steps * 10
        self.keep_checkpoint_max = 5

        # pretrained model path
        self.pretrained_model_path = 'pretrained/vinai/phobert-base'

        # TPU settings
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_job_name = None
        self.tpu_zone = None
        self.gcp_project = None

        model_name_specific_params = {
            'checkpoint_dir', 'log_dir'}
        model_name_specific_kwargs = {}
        for param in model_name_specific_params:
            if param in kwargs:
                model_name_specific_kwargs[param] = kwargs.pop(param)

        self.override_defaults(**kwargs)
        try:
            self.save_checkpoint_freq = int(self.save_checkpoint_freq)
        except:
            assert self.save_checkpoint_freq == 'epoch', \
                "Only `epoch` or integers are supported for `save_checkpoint_freq`. Current value: {}".format(
                    self.save_checkpoint_freq)
        with tf.io.gfile.GFile(self.pipeline_config_file, "r") as reader:
            pipeline_config = json.load(reader)
        self.pipeline_config = pipeline_config

        # model_name-specific params
        self.checkpoint_dir = os.path.join(
            'checkpoints', self.model_name)
        self.log_dir = os.path.join('logs', self.model_name)

        self.override_defaults(**model_name_specific_kwargs)

        if 'log_dir' in model_name_specific_kwargs:
            self.tensorboard_dir = os.path.join(self.log_dir, self.model_name, 'tensorboard')
            self.config_file = os.path.join(self.log_dir, self.model_name, 'config.json')
        else:
            self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
            self.config_file = os.path.join(self.log_dir, 'config.json')
        if 'checkpoint_dir' in model_name_specific_kwargs:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name)
