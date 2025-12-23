import numpy as np
import os

# Limit TensorFlow memory usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode to reduce memory usage

if __name__ == '__main__':
    print("="*70)
    print("QEM HACKATHON SOLUTION - QUICKSTART")
    print("="*70)

    print("\n[1/5] Importing modules...")
    try:
        from qem_data_generator import QEMDataGenerator
        from qem_models import create_qem_model
        from qem_training import QEMTrainer
        print("✓ All modules imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        print("\nPlease install dependencies:")
        print("  pip install qiskit qiskit-aer tensorflow numpy matplotlib scikit-learn --break-system-packages")
        exit(1)

    print("\n[2/5] Generating small quantum dataset...")
    print("  (3 qubits, 5 circuits, 2 depths, 2 noise levels)")

    generator = QEMDataGenerator(num_qubits=3, shots=2048)

    dataset = generator.generate_dataset(
        num_circuits=5,
        depths=[2, 4],
        noise_types=['depolarizing'],
        noise_strengths=[0.05, 0.1],
        circuit_type='random',
        feature_type='bitstring'
    )

    print(f"✓ Generated {len(dataset['x_noisy'])} samples")
    print(f"  Input dimension: {dataset['x_noisy'].shape[1]}")

    print("\n[3/5] Creating adaptive QEM model...")

    input_dim = dataset['x_noisy'].shape[1]
    output_dim = dataset['x_ideal'].shape[1]

    model = create_qem_model(
        'adaptive',
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=8,
        encoder_hidden=[64, 32],
        mitigation_hidden=[128, 64]
    )

    print("✓ Model created")

    print("\n[4/5] Training model...")
    print("  (This will take ~1-2 minutes)")

    trainer = QEMTrainer(
        model=model,
        loss_type='l2',
        learning_rate=1e-3
    )

    from sklearn.model_selection import train_test_split

    x_noisy = dataset['x_noisy']
    x_ideal = dataset['x_ideal']

    x_train, x_test, y_train, y_test = train_test_split(
        x_noisy, x_ideal, test_size=0.2, random_state=42
    )

    history = trainer.model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=8,
        verbose=0,
        validation_split=0.2
    )

    print(f"✓ Training complete")
    print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")

    print("\n[5/5] Evaluating on test set...")

    y_pred = trainer.model.predict(x_test, verbose=0)

    model_mae = np.mean(np.abs(y_pred - y_test))
    baseline_mae = np.mean(np.abs(x_test - y_test))
    improvement = baseline_mae / model_mae
    error_reduction = (baseline_mae - model_mae) / baseline_mae * 100

    print(f"✓ Evaluation complete")
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Baseline (noisy) MAE:     {baseline_mae:.6f}")
    print(f"QEM Model MAE:            {model_mae:.6f}")
    print(f"Improvement Ratio:        {improvement:.2f}x")
    print(f"Error Reduction:          {error_reduction:.1f}%")
    print(f"{'='*70}")

    print("\n✅ QUICKSTART SUCCESSFUL!")
    print("\nNext steps:")
    print("  1. Run 'python qem_examples.py' for comprehensive examples")
    print("  2. See README.md for detailed documentation")
    print("  3. Modify configs to experiment with different settings")
    print("\nHappy hacking! 🚀")
