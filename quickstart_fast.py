import numpy as np
import os
import multiprocessing
import sys

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Limit TensorFlow memory usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode to reduce memory usage

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print("="*70)
    print("QEM HACKATHON SOLUTION - FAST TEST")
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

    print("\n[2/5] Generating improved quantum dataset (FAST)...")
    print("  (3 qubits, 20 circuits, 3 depths, 3 noise levels)")

    generator = QEMDataGenerator(num_qubits=3, shots=8192)

    dataset = generator.generate_dataset(
        num_circuits=20,
        depths=[2, 4, 6],
        noise_types=['depolarizing', 'amplitude_damping'],
        noise_strengths=[0.01, 0.05, 0.1],
        circuit_type='random',
        feature_type='expectation',
        n_jobs=1
    )

    print(f"✓ Generated {len(dataset['x_noisy'])} samples")
    print(f"  Input dimension: {dataset['x_noisy'].shape[1]}")

    print("\n[3/5] Creating improved adaptive QEM model...")

    input_dim = dataset['x_noisy'].shape[1]
    output_dim = dataset['x_ideal'].shape[1]

    model = create_qem_model(
        'adaptive',
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=32,
        encoder_hidden=[256, 128, 64],
        mitigation_hidden=[256, 128, 64],
        use_residual=False
    )

    print("✓ Model created")

    print("\n[4/5] Training model with improved configuration...")
    print("  (This will take ~3-5 minutes)")

    trainer = QEMTrainer(
        model=model,
        loss_type='combined',
        learning_rate=5e-4,
        loss_alpha=0.5
    )

    from sklearn.model_selection import train_test_split
    from tensorflow import keras

    x_noisy = dataset['x_noisy']
    x_ideal = dataset['x_ideal']

    x_train, x_test, y_train, y_test = train_test_split(
        x_noisy, x_ideal, test_size=0.2, random_state=42
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = trainer.model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=64,
        verbose=1,
        validation_split=0.2,
        callbacks=callbacks
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

    print("\n✅ FAST TEST SUCCESSFUL!")
    print("\nFor full training, run 'python quickstart.py'")
