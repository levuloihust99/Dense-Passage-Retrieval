import os
from .common import CommonConfig


class DualEncoderConfig(CommonConfig):
    def __init__(self, **kwargs):
        # dual-encoder-specific
        self.model_name = 'dual-encoder'
        self.model_arch = 'roberta'
        self.context_max_seq_length = 256
        self.query_max_seq_length = 64
        self.use_hardneg = True
        self.use_stratified_loss = True

        # data
        self.tfrecord_dir = 'data/tfrecord/dual_encoder/train'

        # training configurations
        self.learning_rate = 5e-5
        self.train_batch_size = 16
        self.eval_batch_size = 256
        self.num_train_steps = 10000
        self.num_train_epochs = None

        # optimization
        self.lr_decay_power = 1.0 # Linear weight decay by default
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 1000
        self.warmup_proportions = 0.1

        # logging
        self.logging_steps = 100
        self.save_checkpoint_freq = 'epoch'
        self.keep_checkpoint_max = 5

        # pretrained model path
        self.pretrained_model_path = 'pretrained/vinai/phobert-base'

        # TPU settings
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_job_name = None
        self.tpu_zone = None
        self.gcp_project = None

        model_name_specific_params = {'checkpoint_dir', 'tensorboard_dir', 'log_file', 'config_file'}
        model_name_specific_kwargs = {}
        for param in model_name_specific_params:
            if param in kwargs:
                model_name_specific_kwargs[param] = kwargs.pop(param)

        self.override_defaults(**kwargs)
        if self.use_hardneg is False and self.use_stratified_loss is True:
            assert False, "You must use hard negative samples to use stratified loss."
        try:
            self.save_checkpoint_freq = int(self.save_checkpoint_freq)
        except:
            assert self.save_checkpoint_freq == 'epoch', \
                "Only `epoch` or integers are supported for `save_checkpoint_freq`. Current value: {}".format(self.save_checkpoint_freq)

        # model_name-specific params
        self.checkpoint_dir = os.path.join('checkpoints/dual_encoder', self.model_name)
        log_dir = os.path.join('logs/dual_encoder', self.model_name)
        self.tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        self.log_file = os.path.join(log_dir, 'track.log')
        self.config_file = os.path.join(log_dir, 'config.json')

        self.override_defaults(**model_name_specific_kwargs)
