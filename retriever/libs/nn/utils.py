import tensorflow as tf
from libs.nn.configuration import DualEncoderConfig
from libs.data_helpers.constants import (
    TRAIN_MODE, USE_GRADIENT_ACCUMULATE, USE_GRADIENT_CACHE, PIPELINE_SEPERATE_SYMBOL)


# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients 
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices


def validate_config(config: DualEncoderConfig):
    pipeline_config = config.pipeline_config
    available_pipelines = pipeline_config[TRAIN_MODE].split(PIPELINE_SEPERATE_SYMBOL)
    for pipeline_type in available_pipelines:
        if pipeline_config[pipeline_type][USE_GRADIENT_CACHE] \
            and not pipeline_config[pipeline_type][USE_GRADIENT_ACCUMULATE]:
            raise Exception("{}=True and {}=False is not allowed.".format(
                USE_GRADIENT_CACHE, USE_GRADIENT_ACCUMULATE))
