import numpy as np
import os
import sys

from qem_data_generator import QEMDataGenerator
from qem_models import create_qem_model
from qem_training import QEMTrainer, run_training_pipeline


def example_1_quick_start():
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Start with Default Settings")
    print("="*70)
    
    config = {
        'num_qubits': 3,
        'shots': 4096,
        'num_circuits': 20,
        'depths': [2, 4, 6],
        'noise_types': ['depolarizing', 'amplitude_damping'],
        'noise_strengths': [0.01, 0.05, 0.1],
        'model_type': 'adaptive',
        'epochs': 30,
        'batch_size': 32,
        'results_dir': '/home/claude/qem_results_example1'
    }
    
    trainer, results = run_training_pipeline(config)
    
    print("\n✓ Quick start complete!")
    print(f"Improvement ratio: {results['test_metrics']['improvement_ratio']:.2f}x")
    
    return trainer, results


def example_2_explicit_noise_model():
    print("\n" + "="*70)
    print("EXAMPLE 2: Explicit Noise Descriptor Model")
    print("="*70)
    
    config = {
        'num_qubits': 3,
        'shots': 8192,
        'num_circuits': 25,
        'depths': [2, 4, 6, 8],
        'noise_types': ['depolarizing', 'amplitude_damping', 'readout', 'mixed'],
        'noise_strengths': [0.01, 0.03, 0.05, 0.08, 0.1],
        'model_type': 'explicit',
        'loss_type': 'combined',
        'loss_alpha': 0.6,
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 5e-4,
        'model_kwargs': {
            'hidden_dims': [512, 256, 128, 64],
            'use_residual': True
        },
        'results_dir': '/home/claude/qem_results_example2'
    }
    
    trainer, results = run_training_pipeline(config)
    
    print("\n✓ Explicit noise model complete!")
    print(f"Improvement ratio: {results['test_metrics']['improvement_ratio']:.2f}x")
    
    return trainer, results


def example_3_transformer_architecture():
    print("\n" + "="*70)
    print("EXAMPLE 3: Transformer-Based QEM Model")
    print("="*70)
    
    config = {
        'num_qubits': 4,
        'shots': 8192,
        'num_circuits': 20,
        'depths': [3, 5, 7],
        'noise_types': ['depolarizing', 'mixed'],
        'noise_strengths': [0.02, 0.05, 0.1],
        'circuit_type': 'variational',
        'feature_type': 'bitstring',
        'model_type': 'transformer',
        'loss_type': 'l2',
        'epochs': 40,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'model_kwargs': {
            'num_heads': 4,
            'ff_dim': 128,
            'num_transformer_blocks': 2,
            'dropout_rate': 0.1
        },
        'results_dir': '/home/claude/qem_results_example3'
    }
    
    trainer, results = run_training_pipeline(config)
    
    print("\n✓ Transformer model complete!")
    print(f"Improvement ratio: {results['test_metrics']['improvement_ratio']:.2f}x")
    
    return trainer, results


def example_4_expectation_values():
    print("\n" + "="*70)
    print("EXAMPLE 4: Expectation Value QEM")
    print("="*70)
    
    config = {
        'num_qubits': 3,
        'shots': 8192,
        'num_circuits': 30,
        'depths': [2, 4, 6, 8, 10],
        'noise_types': ['depolarizing', 'amplitude_damping', 'mixed'],
        'noise_strengths': [0.01, 0.05, 0.1, 0.15],
        'circuit_type': 'random',
        'feature_type': 'expectation',
        'model_type': 'adaptive',
        'loss_type': 'l2',
        'epochs': 50,
        'batch_size': 32,
        'model_kwargs': {
            'latent_dim': 8,
            'encoder_hidden': [64, 32],
            'mitigation_hidden': [128, 64, 32]
        },
        'results_dir': '/home/claude/qem_results_example4'
    }
    
    trainer, results = run_training_pipeline(config)
    
    print("\n✓ Expectation value QEM complete!")
    print(f"Improvement ratio: {results['test_metrics']['improvement_ratio']:.2f}x")
    
    return trainer, results


def example_5_comprehensive_benchmark():
    print("\n" + "="*70)
    print("EXAMPLE 5: Comprehensive Benchmark Suite")
    print("="*70)

    print("\nGenerating comprehensive dataset...")
    generator = QEMDataGenerator(num_qubits=3, shots=8192)
    
    dataset = generator.generate_dataset(
        num_circuits=50,
        depths=[2, 4, 6, 8, 10, 12],
        noise_types=['depolarizing', 'amplitude_damping', 'readout', 'mixed'],
        noise_strengths=[0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
        circuit_type='random',
        feature_type='bitstring'
    )
    
    dataset_path = '/home/claude/qem_comprehensive_dataset.npz'
    generator.save_dataset(dataset, dataset_path)

    config_adaptive = {
        'dataset_path': dataset_path,
        'model_type': 'adaptive',
        'loss_type': 'combined',
        'loss_alpha': 0.5,
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'early_stopping_patience': 20,
        'model_kwargs': {
            'latent_dim': 16,
            'encoder_hidden': [128, 64],
            'mitigation_hidden': [256, 128, 64]
        },
        'results_dir': '/home/claude/qem_results_comprehensive'
    }
    
    trainer, results = run_training_pipeline(config_adaptive)
    
    print("\n✓ Comprehensive benchmark complete!")
    print("\nPerformance Summary:")
    print(f"  Overall improvement: {results['test_metrics']['improvement_ratio']:.2f}x")
    print(f"  Error reduction: {results['test_metrics']['error_reduction']:.2f}%")

    print("\nBreakdown by Noise Type:")
    for noise_type, metrics in results['benchmark_results']['by_noise_type'].items():
        print(f"  {noise_type}: {metrics['improvement_ratio']:.2f}x improvement")
    
    return trainer, results


def compare_models():
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    print("\nGenerating shared dataset for comparison...")
    generator = QEMDataGenerator(num_qubits=3, shots=8192)
    
    dataset = generator.generate_dataset(
        num_circuits=30,
        depths=[2, 4, 6, 8],
        noise_types=['depolarizing', 'amplitude_damping', 'mixed'],
        noise_strengths=[0.01, 0.05, 0.1],
        circuit_type='random',
        feature_type='bitstring'
    )
    
    dataset_path = '/home/claude/qem_comparison_dataset.npz'
    generator.save_dataset(dataset, dataset_path)

    results_comparison = {}

    models_to_test = [
        ('adaptive', {'latent_dim': 16}),
        ('explicit', {}),
        ('transformer', {'num_heads': 4, 'ff_dim': 128})
    ]
    
    for model_type, model_kwargs in models_to_test:
        print(f"\n--- Testing {model_type.upper()} model ---")
        
        config = {
            'dataset_path': dataset_path,
            'model_type': model_type,
            'model_kwargs': model_kwargs,
            'loss_type': 'l2',
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'verbose': 0,
            'results_dir': f'/home/claude/qem_results_comparison_{model_type}'
        }
        
        trainer, results = run_training_pipeline(config)
        
        results_comparison[model_type] = {
            'improvement_ratio': results['test_metrics']['improvement_ratio'],
            'model_mae': results['test_metrics']['model_mae'],
            'error_reduction': results['test_metrics']['error_reduction']
        }

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Model':<15} {'Improvement':<15} {'MAE':<15} {'Error Reduction':<15}")
    print("-" * 70)
    
    for model_type, metrics in results_comparison.items():
        print(f"{model_type:<15} "
              f"{metrics['improvement_ratio']:.2f}x{' ':<11} "
              f"{metrics['model_mae']:.6f}{' ':<6} "
              f"{metrics['error_reduction']:.2f}%")
    
    return results_comparison


def custom_experiment():
    print("\n" + "="*70)
    print("CUSTOM EXPERIMENT TEMPLATE")
    print("="*70)
    print("\nModify this function to run your custom experiments!")

    config = {
        'num_qubits': 3,
        'shots': 8192,
        'num_circuits': 20,
        'depths': [2, 4, 6],
        'noise_types': ['depolarizing'],
        'noise_strengths': [0.05],
        'model_type': 'adaptive',
        'epochs': 30,
        'batch_size': 32,
        'results_dir': '/home/claude/qem_results_custom'
    }
    
    trainer, results = run_training_pipeline(config)
    
    return trainer, results


def main():
    print("\n" + "="*70)
    print("QEM HACKATHON SOLUTION - EXAMPLE WORKFLOWS")
    print("="*70)
    print("\nAvailable examples:")
    print("1. Quick start with default settings")
    print("2. Explicit noise descriptor model")
    print("3. Transformer-based architecture")
    print("4. Expectation value QEM")
    print("5. Comprehensive benchmark suite")
    print("6. Model comparison")
    print("7. Custom experiment template")


    print("\n" + "="*70)
    print("Choose which examples to run by uncommenting them in the main() function")
    print("="*70)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("\nRunning a simple demonstration...")
    print("(Uncomment other examples in main() for more)")

    config = {
        'num_qubits': 3,
        'shots': 4096,
        'num_circuits': 10,
        'depths': [2, 4],
        'noise_types': ['depolarizing'],
        'noise_strengths': [0.05, 0.1],
        'model_type': 'adaptive',
        'epochs': 20,
        'batch_size': 32,
        'verbose': 1,
        'results_dir': 'qem_demo',
        'n_jobs': 1
    }

    trainer, results = run_training_pipeline(config)

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {config['results_dir']}")
    print(f"Improvement ratio: {results['test_metrics']['improvement_ratio']:.2f}x")
    print(f"Error reduction: {results['test_metrics']['error_reduction']:.2f}%")
    print("\nFor more comprehensive experiments, uncomment examples in main()")
