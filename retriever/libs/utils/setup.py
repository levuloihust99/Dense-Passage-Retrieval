import tensorflow as tf


def setup_distribute_strategy(use_tpu=False, tpu_name=''):
    if use_tpu:
        use_gpu = False
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name) # TPU detection
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
        except Exception:
            use_gpu = True
    else:
        use_gpu = True

    if use_gpu: # detect GPUs
        devices = tf.config.list_physical_devices("GPU")
        if devices:
            use_default_strategy = False
            strategy = tf.distribute.MirroredStrategy()
        else:
            use_default_strategy = True
    else:
        use_default_strategy = False

    if use_default_strategy:
        strategy = tf.distribute.get_strategy()

    return strategy


def setup_memory_growth():
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)


def setup_random(seed):
    if seed is not None:
        tf.random.set_seed(seed)
