"""Compatible positional embedding for the author's saved Keras model.

Reference notebook and commit are recorded in configs/wisdm-model.json.
This module does not load or run a model on import.
"""
from keras import layers, ops


class AddPositionEmbedding(layers.Layer):
    """Add learned temporal positions, keeping the saved layer's weight layout."""

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        # The attribute name matches the saved model's nested embedding layer.
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def build(self, input_shape):
        if not self.pos_emb.built:
            self.pos_emb.build((input_shape[1],))
        super().build(input_shape)

    def call(self, inputs):
        positions = ops.arange(ops.shape(inputs)[1], dtype="int32")
        return inputs + ops.expand_dims(self.pos_emb(positions), axis=0)

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len, "d_model": self.d_model}
