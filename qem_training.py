import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional
import json
import os

from qem_models import create_qem_model, QEMLoss, mean_absolute_error_metric
from qem_data_generator import QEMDataGenerator


class QEMTrainer:

    def __init__(self,
                 model,
                 loss_type: str = 'l2',
                 learning_rate: float = 1e-3,
                 loss_alpha: float = 0.5):
        self.model = model
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha

        if loss_type == 'l1':
            self.loss_fn = QEMLoss.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = QEMLoss.l2_loss
        elif loss_type == 'fidelity':
            self.loss_fn = QEMLoss.fidelity_loss
        elif loss_type == 'combined':
            self.loss_fn = lambda y_true, y_pred: QEMLoss.combined_loss(
                y_true, y_pred, alpha=loss_alpha
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=[mean_absolute_error_metric]
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

    def prepare_data(self,
                     dataset: Dict,
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     use_explicit_noise: bool = False) -> Tuple:
        x_noisy = dataset['x_noisy']
        x_ideal = dataset['x_ideal']

        x_noisy_train, x_noisy_test, x_ideal_train, x_ideal_test = train_test_split(
            x_noisy, x_ideal, test_size=test_size, random_state=42
        )

        x_noisy_train, x_noisy_val, x_ideal_train, x_ideal_val = train_test_split(
            x_noisy_train, x_ideal_train, test_size=val_size, random_state=42
        )

        if use_explicit_noise:
            noise_desc = dataset['noise_descriptors']
            
            noise_desc_train, noise_desc_test = train_test_split(
                noise_desc, test_size=test_size, random_state=42
            )
            noise_desc_train, noise_desc_val = train_test_split(
                noise_desc_train, test_size=val_size, random_state=42
            )
            
            train_data = ([x_noisy_train, noise_desc_train], x_ideal_train)
            val_data = ([x_noisy_val, noise_desc_val], x_ideal_val)
            test_data = ([x_noisy_test, noise_desc_test], x_ideal_test)
        else:
            train_data = (x_noisy_train, x_ideal_train)
            val_data = (x_noisy_val, x_ideal_val)
            test_data = (x_noisy_test, x_ideal_test)
        
        print(f"Data split:")
        print(f"  Training samples: {len(x_ideal_train)}")
        print(f"  Validation samples: {len(x_ideal_val)}")
        print(f"  Test samples: {len(x_ideal_test)}")
        
        return train_data, val_data, test_data
    
    def train(self,
              train_data: Tuple,
              val_data: Tuple,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1,
              early_stopping_patience: int = 15) -> Dict:
        x_train, y_train = train_data
        x_val, y_val = val_data

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        self.history['train_loss'] = history.history['loss']
        self.history['val_loss'] = history.history['val_loss']
        self.history['train_mae'] = history.history['mean_absolute_error_metric']
        self.history['val_mae'] = history.history['val_mean_absolute_error_metric']
        
        return self.history
    
    def evaluate(self, test_data: Tuple, baseline_method: str = 'noisy') -> Dict:
        x_test, y_test = test_data

        if isinstance(x_test, list):
            y_pred = self.model.predict(x_test)
            x_noisy = x_test[0]
        else:
            y_pred = self.model.predict(x_test)
            x_noisy = x_test

        model_error = np.abs(y_pred - y_test)
        baseline_error = np.abs(x_noisy - y_test)

        metrics = {
            'model_mae': np.mean(model_error),
            'model_max_error': np.max(model_error),
            'model_mse': np.mean((y_pred - y_test) ** 2),

            'baseline_mae': np.mean(baseline_error),
            'baseline_max_error': np.max(baseline_error),
            'baseline_mse': np.mean((x_noisy - y_test) ** 2),

            'improvement_ratio': np.mean(baseline_error) / np.mean(model_error),
            'error_reduction': (np.mean(baseline_error) - np.mean(model_error)) / np.mean(baseline_error) * 100,

            'per_sample_mae': np.mean(model_error, axis=1),
            'per_sample_baseline_mae': np.mean(baseline_error, axis=1)
        }
        
        return metrics
    
    def benchmark_across_noise(self,
                               dataset: Dict,
                               use_explicit_noise: bool = False) -> Dict:
        results = {
            'by_noise_strength': {},
            'by_depth': {},
            'by_noise_type': {}
        }
        
        x_noisy = dataset['x_noisy']
        x_ideal = dataset['x_ideal']
        noise_strengths = dataset['noise_strengths']
        depths = dataset['circuit_depths']
        noise_types = dataset['noise_types']

        if use_explicit_noise:
            noise_desc = dataset['noise_descriptors']
            inputs = [x_noisy, noise_desc]
        else:
            inputs = x_noisy

        y_pred = self.model.predict(inputs)

        unique_strengths = np.unique(noise_strengths)
        for strength in unique_strengths:
            mask = noise_strengths == strength

            if np.sum(mask) == 0:
                continue

            model_error = np.mean(np.abs(y_pred[mask] - x_ideal[mask]))
            baseline_error = np.mean(np.abs(x_noisy[mask] - x_ideal[mask]))

            results['by_noise_strength'][float(strength)] = {
                'model_mae': float(model_error),
                'baseline_mae': float(baseline_error),
                'improvement_ratio': float(baseline_error / model_error) if model_error > 0 else float('inf')
            }

        unique_depths = np.unique(depths)
        for depth in unique_depths:
            mask = depths == depth

            if np.sum(mask) == 0:
                continue

            model_error = np.mean(np.abs(y_pred[mask] - x_ideal[mask]))
            baseline_error = np.mean(np.abs(x_noisy[mask] - x_ideal[mask]))

            results['by_depth'][int(depth)] = {
                'model_mae': float(model_error),
                'baseline_mae': float(baseline_error),
                'improvement_ratio': float(baseline_error / model_error) if model_error > 0 else float('inf')
            }

        unique_types = np.unique(noise_types)
        for noise_type in unique_types:
            mask = noise_types == noise_type

            if np.sum(mask) == 0:
                continue

            model_error = np.mean(np.abs(y_pred[mask] - x_ideal[mask]))
            baseline_error = np.mean(np.abs(x_noisy[mask] - x_ideal[mask]))

            results['by_noise_type'][str(noise_type)] = {
                'model_mae': float(model_error),
                'baseline_mae': float(baseline_error),
                'improvement_ratio': float(baseline_error / model_error) if model_error > 0 else float('inf')
            }
        
        return results

    def plot_training_history(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['train_mae'], label='Train MAE')
        axes[1].plot(self.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        return fig

    def plot_benchmark_results(self, benchmark_results: Dict, save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        strengths = sorted(benchmark_results['by_noise_strength'].keys())
        model_mae = [benchmark_results['by_noise_strength'][s]['model_mae'] for s in strengths]
        baseline_mae = [benchmark_results['by_noise_strength'][s]['baseline_mae'] for s in strengths]
        
        ax.plot(strengths, baseline_mae, 'o-', label='Baseline (noisy)', linewidth=2)
        ax.plot(strengths, model_mae, 's-', label='QEM Model', linewidth=2)
        ax.set_xlabel('Noise Strength')
        ax.set_ylabel('MAE')
        ax.set_title('Performance vs Noise Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        depths = sorted(benchmark_results['by_depth'].keys())
        model_mae = [benchmark_results['by_depth'][d]['model_mae'] for d in depths]
        baseline_mae = [benchmark_results['by_depth'][d]['baseline_mae'] for d in depths]
        
        ax.plot(depths, baseline_mae, 'o-', label='Baseline (noisy)', linewidth=2)
        ax.plot(depths, model_mae, 's-', label='QEM Model', linewidth=2)
        ax.set_xlabel('Circuit Depth')
        ax.set_ylabel('MAE')
        ax.set_title('Performance vs Circuit Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        noise_types = list(benchmark_results['by_noise_type'].keys())
        improvement_ratios = [benchmark_results['by_noise_type'][nt]['improvement_ratio'] for nt in noise_types]
        
        ax.bar(noise_types, improvement_ratios, alpha=0.7)
        ax.axhline(y=1.0, color='r', linestyle='--', label='No improvement')
        ax.set_xlabel('Noise Type')
        ax.set_ylabel('Improvement Ratio')
        ax.set_title('Improvement Ratio by Noise Type')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Benchmark plot saved to {save_path}")
        
        return fig

    def save_model(self, filepath: str):
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_model(self, filepath: str):
        self.model.load_weights(filepath)
        print(f"Model weights loaded from {filepath}")


def run_training_pipeline(config: Dict) -> Tuple[QEMTrainer, Dict]:
    print("=" * 70)
    print("QEM Training Pipeline")
    print("=" * 70)

    print("\n1. Loading dataset...")
    generator = QEMDataGenerator(
        num_qubits=config['num_qubits'],
        shots=config['shots']
    )
    
    if 'dataset_path' in config and os.path.exists(config['dataset_path']):
        dataset = generator.load_dataset(config['dataset_path'])
    else:
        dataset = generator.generate_dataset(
            num_circuits=config['num_circuits'],
            depths=config['depths'],
            noise_types=config['noise_types'],
            noise_strengths=config['noise_strengths'],
            circuit_type=config.get('circuit_type', 'random'),
            feature_type=config.get('feature_type', 'bitstring'),
            n_jobs=config.get('n_jobs', -1)
        )
        if 'dataset_path' in config:
            generator.save_dataset(dataset, config['dataset_path'])

    print("\n2. Creating model...")
    input_dim = dataset['x_noisy'].shape[1]
    output_dim = dataset['x_ideal'].shape[1]
    
    model = create_qem_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        output_dim=output_dim,
        noise_descriptor_dim=dataset['noise_descriptors'].shape[1] if config['model_type'] == 'explicit' else None,
        **config.get('model_kwargs', {})
    )
    
    print(f"Model type: {config['model_type']}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")

    print("\n3. Setting up trainer...")
    trainer = QEMTrainer(
        model=model,
        loss_type=config.get('loss_type', 'l2'),
        learning_rate=config.get('learning_rate', 1e-3),
        loss_alpha=config.get('loss_alpha', 0.5)
    )

    use_explicit_noise = (config['model_type'] == 'explicit')
    train_data, val_data, test_data = trainer.prepare_data(
        dataset,
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.1),
        use_explicit_noise=use_explicit_noise
    )

    print("\n4. Training model...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        verbose=config.get('verbose', 1),
        early_stopping_patience=config.get('early_stopping_patience', 15)
    )

    print("\n5. Evaluating model...")
    test_metrics = trainer.evaluate(test_data)
    
    print("\nTest Set Results:")
    print(f"  Model MAE: {test_metrics['model_mae']:.6f}")
    print(f"  Baseline MAE: {test_metrics['baseline_mae']:.6f}")
    print(f"  Improvement Ratio: {test_metrics['improvement_ratio']:.2f}x")
    print(f"  Error Reduction: {test_metrics['error_reduction']:.2f}%")

    print("\n6. Benchmarking across conditions...")
    benchmark_results = trainer.benchmark_across_noise(dataset, use_explicit_noise)

    if 'results_dir' in config:
        os.makedirs(config['results_dir'], exist_ok=True)

        trainer.plot_training_history(
            os.path.join(config['results_dir'], 'training_history.png')
        )
        trainer.plot_benchmark_results(
            benchmark_results,
            os.path.join(config['results_dir'], 'benchmark_results.png')
        )

        trainer.save_model(os.path.join(config['results_dir'], 'model_weights.weights.h5'))

        with open(os.path.join(config['results_dir'], 'test_metrics.json'), 'w') as f:
            metrics_serializable = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in test_metrics.items()
                if not isinstance(v, np.ndarray)
            }
            json.dump(metrics_serializable, f, indent=2)

        with open(os.path.join(config['results_dir'], 'benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)

        print(f"\nResults saved to {config['results_dir']}")

    print("\n" + "=" * 70)
    print("Training pipeline complete!")
    print("=" * 70)
    
    return trainer, {'test_metrics': test_metrics, 'benchmark_results': benchmark_results}


if __name__ == "__main__":
    config = {
        'num_qubits': 3,
        'shots': 8192,
        'num_circuits': 10,
        'depths': [2, 4, 6],
        'noise_types': ['depolarizing', 'amplitude_damping'],
        'noise_strengths': [0.01, 0.05, 0.1],
        'circuit_type': 'random',
        'feature_type': 'bitstring',
        'model_type': 'adaptive',
        'loss_type': 'l2',
        'learning_rate': 1e-3,
        'epochs': 50,
        'batch_size': 32,
        'test_size': 0.2,
        'val_size': 0.1,
        'verbose': 1,
        'dataset_path': 'qem_dataset_example.npz',
        'results_dir': 'qem_results_example'
    }

    trainer, results = run_training_pipeline(config)
