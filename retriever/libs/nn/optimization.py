import tensorflow as tf

from official import nlp
import official.nlp.optimization

import math
from typing import Dict


def get_adamw(
    num_train_steps: int,
    warmup_steps: int,
    learning_rate: float,
    weight_decay_rate: float = 0.01,
    eps: float = 1e-6,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    gradient_clip_norm: float = 1.0
):
    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=0
    )

    warmup_schedule = nlp.optimization.WarmUp(
        initial_learning_rate=learning_rate,
        decay_schedule_fn=decay_schedule,
        warmup_steps=warmup_steps
    )

    return nlp.optimization.AdamWeightDecay(
        learning_rate=warmup_schedule,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=eps,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'],
        gradient_clip_norm=-1.0, # this is a bug in tf-models-official 2.4.0
        global_clipnorm=gradient_clip_norm
    )
