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
