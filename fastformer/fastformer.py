import numpy as np
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, input_vocab_size, **kwargs):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.max_position_encoding = 10000

    def build(self, input_shape):
        self.emb_layer = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)
        self.pos_layer = PositionalEncodingLayer(self.d_model, self.max_position_encoding)
        self.attention_layer = SelfAttention(self.d_model)
        self.feed_forward_layer = Feedforward(self.d_model)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.emb_layer(inputs)
        x = self.pos_layer(x)
        x = self.attention_layer(x)
        x = self.feed_forward_layer(x)
        return x


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, max_position_encoding, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_position_encoding = max_position_encoding

    def build(self, input_shape):
        self.mult = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()
        super().build(input_shape)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], self.d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        ## positional encoding
        scaling_factor = tf.constant(np.sqrt(self.d_model), shape=(1, 1, 1))
        x = self.mult([x, scaling_factor])
        pos = self.positional_encoding(self.maximum_position_encoding, self.d_model)
        return self.add([x, pos[:, :tf.shape(x)[1], :]])

class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, use_scale=True, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.use_scale = use_scale

    def build(self, input_shape):
        v_dim = tf.TensorShape(input_shape[1])[-1]
        self.scale = None
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=[v_dim],
                initializer="glorot_uniform",
                dtype=self.dtype,
                trainable=True,
            )


        super().build(input_shape)
        self.built = True

    def _calculate_scores(self, query, key):
        q_reshaped = tf.expand_dims(query, axis=-2)
        k_reshaped = tf.expand_dims(key, axis=-3)
        return tf.reduce_sum(self.scale * tf.tanh(q_reshaped + k_reshaped), axis=-1)

    def _apply_scores(self, scores, value):
        weights = tf.nn.softmax(scores)
        return tf.matmul(weights, value), weights

    def _validate_call_args(self, inputs):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                f"{class_name} layer must be called on a list of inputs, "
                "namely [query, value] or [query, value, key]. "
                f"Received: {inputs}."
            )
        if len(inputs) < 2 or len(inputs) > 3:
            raise ValueError(
                f"{class_name} layer accepts inputs list of length 2 or 3, "
                "namely [query, value] or [query, value, key]. "
                f"Received length: {len(inputs)}."
            )

    def call(self, inputs):
        self._validate_call_args(inputs=inputs)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        scores = self._calculate_scores(query=q, key=k)
        result, attention_scores = self._apply_scores(scores=scores, value=v)
        return result

    def get_config(self):
        config = {
            "dropout": self.dropout,
            "use_scale": self.use_scale,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))




class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        self.d_model = d_model

    def build(self, input_shape):
        self.query = tf.keras.layers.Dense(self.d_model)
        self.key = tf.keras.layers.Dense(self.d_model)
        self.value = tf.keras.layers.Dense(self.d_model)
        self.attention_layer = AdditiveAttention()
        self.dense = tf.keras.layers.Dense(self.d_model)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)


    def call(self, x, **kwargs):
        query = self.query(x)
        value = self.key(x)
        key = self.value(x)
        attention = self.attention_layer([query, value, key])
        attention = self.dense(attention)
        x = self.add([x, attention])  # residual connection
        x = self.layer_norm(x)
        return x


class Feedforward(tf.keras.layers.Layer):
    def __init__(self, d_model=512, d_ff=2048, **kwargs):
        super(Feedforward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

    def call(self, inputs, **kwargs):
        dense = tf.keras.layers.Dense(self.d_ff, activation='relu')(inputs)
        dense = tf.keras.layers.Dense(self.d_model)(dense)
        x = tf.keras.layers.Add()([inputs, dense])  # residual connection
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
