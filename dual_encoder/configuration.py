import os
import json
import tensorflow as tf


class DualEncoderConfig(object):
    def __init__(self, **kwargs):
        self.model_name = 'dualencoder'
        self.model_arch = 'distilbert'
        self.debug = False
        self.query_max_seq_length = 32
        self.context_max_seq_length = 256

        # training configurations
        self.train_batch_size = 16
        self.eval_batch_size = 256
        self.num_train_steps = 10000
        self.num_train_epochs = None

        # optimization
        self.learning_rate = 5e-5
        self.lr_decay_power = 1.0 # Linear weight decay by default
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 1000
        self.warmup_proportions = 0.1

        # logging
        self.logging_steps = 100
        self.save_checkpoint_freq = 'epoch'
        self.keep_checkpoint_max = 5

        # default locations
        self.data_dir = 'data'
        self.tokenizer_path = 'pretrained/distilbert-cmc-A8-H512-L4'
        self.pretrained_model_path = 'pretrained/distilbert-cmc-A8-H512-L4'
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'

        # TPU settings
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_job_name = None
        self.tpu_zone = None
        self.gcp_project = None

        self.checkpoint_path = None
        self.data_tfrecord_dir = None
        self.log_path = None

        self.update(**kwargs)
        try:
            self.save_checkpoint_freq = int(self.save_checkpoint_freq)
        except:
            assert self.save_checkpoint_freq == 'epoch', \
                "Only `epoch` or integers are supported for `save_checkpoint_freq`. Current value: {}".format(self.save_checkpoint_freq)

        # derivatived configurations
        self.data_tfrecord_dir = os.path.join(self.data_dir, 'tfrecord', 'train')
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_name)
        self.log_path = os.path.join(self.log_dir, self.model_name)

        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise ValueError("Unknown hparam " + k)
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
