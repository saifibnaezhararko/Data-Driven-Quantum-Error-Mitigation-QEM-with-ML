import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class NoiseEncoderNetwork(Model):

    def __init__(self, latent_dim: int = 16, hidden_dims: List[int] = [128, 64]):
        super(NoiseEncoderNetwork, self).__init__()
        
        self.encoder_layers = []
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.encoder_layers.append(layers.BatchNormalization())
            self.encoder_layers.append(layers.Dropout(0.1))
        
        self.latent_layer = layers.Dense(latent_dim, activation='tanh', name='latent_noise')

    def call(self, x, training=False):
        h = x
        for layer in self.encoder_layers:
            h = layer(h, training=training)
        z = self.latent_layer(h)
        return z


class ConditionalMitigationNetwork(Model):

    def __init__(self, output_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(ConditionalMitigationNetwork, self).__init__()
        
        self.mitigation_layers = []
        for hidden_dim in hidden_dims:
            self.mitigation_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.mitigation_layers.append(layers.BatchNormalization())
            self.mitigation_layers.append(layers.Dropout(0.1))
        
        self.output_layer = layers.Dense(output_dim, activation='linear', name='mitigated_output')

    def call(self, inputs, training=False):
        x_n, z = inputs

        h = tf.concat([x_n, z], axis=-1)
        
        for layer in self.mitigation_layers:
            h = layer(h, training=training)
        
        x_hat = self.output_layer(h)
        return x_hat


class AdaptiveQEMModel(Model):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int = 16,
                 encoder_hidden: List[int] = [128, 64],
                 mitigation_hidden: List[int] = [256, 128, 64],
                 use_residual: bool = False):
        super(AdaptiveQEMModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        self.noise_encoder = NoiseEncoderNetwork(latent_dim, encoder_hidden)

        self.mitigation_network = ConditionalMitigationNetwork(output_dim, mitigation_hidden)

    def call(self, x_noisy, training=False):

        z = self.noise_encoder(x_noisy, training=training)

        x_mitigated = self.mitigation_network([x_noisy, z], training=training)

        if self.use_residual:
            x_mitigated = x_noisy + x_mitigated
        
        return x_mitigated

    def get_noise_descriptor(self, x_noisy):
        return self.noise_encoder(x_noisy, training=False)


class ExplicitNoiseQEMModel(Model):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 noise_descriptor_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 use_residual: bool = False):
        super(ExplicitNoiseQEMModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        self.mitigation_layers = []

        input_size = input_dim + noise_descriptor_dim
        
        for hidden_dim in hidden_dims:
            self.mitigation_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.mitigation_layers.append(layers.BatchNormalization())
            self.mitigation_layers.append(layers.Dropout(0.1))
        
        self.output_layer = layers.Dense(output_dim, activation='linear')

    def call(self, inputs, training=False):
        x_noisy, noise_descriptor = inputs

        h = tf.concat([x_noisy, noise_descriptor], axis=-1)

        for layer in self.mitigation_layers:
            h = layer(h, training=training)

        x_mitigated = self.output_layer(h)

        if self.use_residual:
            x_mitigated = x_noisy + x_mitigated
        
        return x_mitigated


class TransformerQEMModel(Model):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 ff_dim: int = 128,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.1):
        super(TransformerQEMModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_projection = layers.Dense(ff_dim)

        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(num_heads, ff_dim, dropout_rate)
            )

        self.output_dense = layers.Dense(ff_dim, activation='relu')
        self.output_layer = layers.Dense(output_dim, activation='linear')

    def call(self, x_noisy, training=False):

        x = tf.expand_dims(x_noisy, axis=1)

        x = self.input_projection(x)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = tf.squeeze(x, axis=1)
        x = self.output_dense(x)
        x_mitigated = self.output_layer(x)

        x_mitigated = x_noisy + x_mitigated
        
        return x_mitigated


class TransformerBlock(layers.Layer):

    def __init__(self, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ff_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim * 2, activation='relu'),
            layers.Dense(ff_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def create_qem_model(model_type: str,
                     input_dim: int,
                     output_dim: int,
                     noise_descriptor_dim: Optional[int] = None,
                     **kwargs) -> Model:

    if model_type == 'adaptive':
        model = AdaptiveQEMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=kwargs.get('latent_dim', 16),
            encoder_hidden=kwargs.get('encoder_hidden', [128, 64]),
            mitigation_hidden=kwargs.get('mitigation_hidden', [256, 128, 64]),
            use_residual=kwargs.get('use_residual', False)
        )

    elif model_type == 'explicit':
        if noise_descriptor_dim is None:
            raise ValueError("noise_descriptor_dim required for explicit model")
        
        model = ExplicitNoiseQEMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            noise_descriptor_dim=noise_descriptor_dim,
            hidden_dims=kwargs.get('hidden_dims', [256, 128, 64]),
            use_residual=kwargs.get('use_residual', False)
        )

    elif model_type == 'transformer':
        model = TransformerQEMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=kwargs.get('num_heads', 4),
            ff_dim=kwargs.get('ff_dim', 128),
            num_transformer_blocks=kwargs.get('num_transformer_blocks', 2),
            dropout_rate=kwargs.get('dropout_rate', 0.1)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


class QEMLoss:

    @staticmethod
    def l1_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    @staticmethod
    def l2_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def fidelity_loss(y_true, y_pred):
        y_true_pos = tf.maximum(y_true, 1e-10)
        y_pred_pos = tf.maximum(y_pred, 1e-10)

        y_true_norm = y_true_pos / tf.reduce_sum(y_true_pos, axis=-1, keepdims=True)
        y_pred_norm = y_pred_pos / tf.reduce_sum(y_pred_pos, axis=-1, keepdims=True)

        fidelity = tf.square(tf.reduce_sum(tf.sqrt(y_true_norm * y_pred_norm), axis=-1))

        return tf.reduce_mean(1.0 - fidelity)

    @staticmethod
    def combined_loss(y_true, y_pred, alpha=0.5):
        l2 = QEMLoss.l2_loss(y_true, y_pred)
        fid = QEMLoss.fidelity_loss(y_true, y_pred)
        return alpha * l2 + (1 - alpha) * fid


def mean_absolute_error_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def max_absolute_error_metric(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))


if __name__ == "__main__":
    print("QEM Model Architectures - Test")
    print("=" * 50)

    input_dim = 8
    output_dim = 8

    print("\n1. Testing Adaptive QEM Model...")
    adaptive_model = create_qem_model(
        'adaptive',
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=16
    )

    test_input = np.random.rand(4, input_dim).astype(np.float32)
    test_output = adaptive_model(test_input, training=False)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

    print("\n2. Testing Explicit Noise QEM Model...")
    explicit_model = create_qem_model(
        'explicit',
        input_dim=input_dim,
        output_dim=output_dim,
        noise_descriptor_dim=3
    )

    test_noise_desc = np.random.rand(4, 3).astype(np.float32)
    test_output = explicit_model([test_input, test_noise_desc], training=False)
    print(f"Output shape: {test_output.shape}")

    print("\n3. Testing Transformer QEM Model...")
    transformer_model = create_qem_model(
        'transformer',
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=4
    )

    test_output = transformer_model(test_input, training=False)
    print(f"Output shape: {test_output.shape}")

    print("\nAll models created successfully!")
