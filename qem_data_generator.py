import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    ReadoutError
)
from typing import List, Dict, Tuple, Optional
import json
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class QEMDataGenerator:
    def __init__(self, num_qubits: int = 3, shots: int = 8192):
        self.num_qubits = num_qubits
        self.shots = shots
        self.ideal_simulator = AerSimulator(method='statevector')
        self._simulator_cache = {}
        self._transpile_cache = {}
        
    def create_random_circuit(self, depth: int, seed: Optional[int] = None) -> QuantumCircuit:
        if seed is not None:
            np.random.seed(seed)
            
        qc = QuantumCircuit(self.num_qubits)

        gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']
        
        for _ in range(depth):
            gate = np.random.choice(gates)
            
            if gate == 'h':
                qubit = np.random.randint(self.num_qubits)
                qc.h(qubit)
            elif gate == 'x':
                qubit = np.random.randint(self.num_qubits)
                qc.x(qubit)
            elif gate == 'y':
                qubit = np.random.randint(self.num_qubits)
                qc.y(qubit)
            elif gate == 'z':
                qubit = np.random.randint(self.num_qubits)
                qc.z(qubit)
            elif gate in ['rx', 'ry', 'rz']:
                qubit = np.random.randint(self.num_qubits)
                angle = np.random.uniform(0, 2 * np.pi)
                if gate == 'rx':
                    qc.rx(angle, qubit)
                elif gate == 'ry':
                    qc.ry(angle, qubit)
                else:
                    qc.rz(angle, qubit)
            elif gate == 'cx':
                if self.num_qubits > 1:
                    control = np.random.randint(self.num_qubits)
                    target = np.random.randint(self.num_qubits)
                    while target == control:
                        target = np.random.randint(self.num_qubits)
                    qc.cx(control, target)
        
        return qc
    
    def create_variational_circuit(self, depth: int, params: Optional[np.ndarray] = None) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        
        if params is None:
            params = np.random.uniform(0, 2*np.pi, size=depth * self.num_qubits * 2)
        
        param_idx = 0
        for layer in range(depth):
            for qubit in range(self.num_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
                qc.rz(params[param_idx], qubit)
                param_idx += 1

            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)

            if self.num_qubits > 2:
                qc.cx(self.num_qubits - 1, 0)
        
        return qc
    
    def create_noise_model(self, noise_type: str, noise_strength: float) -> NoiseModel:
        noise_model = NoiseModel()
        
        if noise_type == 'depolarizing':
            error_1q = depolarizing_error(noise_strength, 1)
            error_2q = depolarizing_error(noise_strength * 1.5, 2)
            
            noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

        elif noise_type == 'amplitude_damping':
            error_1q = amplitude_damping_error(noise_strength)
            noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])

        elif noise_type == 'readout':
            readout_error = ReadoutError([[1 - noise_strength, noise_strength],
                                         [noise_strength, 1 - noise_strength]])
            noise_model.add_all_qubit_readout_error(readout_error)

        elif noise_type == 'mixed':
            depol_1q = depolarizing_error(noise_strength * 0.5, 1)
            depol_2q = depolarizing_error(noise_strength * 0.75, 2)
            amp_damp = amplitude_damping_error(noise_strength * 0.3)

            combined_1q = depol_1q.compose(amp_damp)
            
            noise_model.add_all_qubit_quantum_error(combined_1q, ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
            noise_model.add_all_qubit_quantum_error(depol_2q, ['cx'])

            readout_error = ReadoutError([[1 - noise_strength * 0.4, noise_strength * 0.4],
                                         [noise_strength * 0.4, 1 - noise_strength * 0.4]])
            noise_model.add_all_qubit_readout_error(readout_error)
        
        return noise_model
    
    def _get_simulator(self, noise_model: Optional[NoiseModel] = None) -> AerSimulator:
        if noise_model is None:
            return self.ideal_simulator

        noise_key = id(noise_model)
        if noise_key not in self._simulator_cache:
            self._simulator_cache[noise_key] = AerSimulator(noise_model=noise_model)
        return self._simulator_cache[noise_key]

    def _get_transpiled_circuit(self, qc: QuantumCircuit, simulator: AerSimulator) -> QuantumCircuit:
        circuit_key = (id(qc), id(simulator))
        if circuit_key not in self._transpile_cache:
            self._transpile_cache[circuit_key] = transpile(qc, simulator)
        return self._transpile_cache[circuit_key]

    def get_bitstring_probabilities(self, qc: QuantumCircuit, noise_model: Optional[NoiseModel] = None) -> np.ndarray:
        qc_measure = qc.copy()
        qc_measure.measure_all()

        simulator = self._get_simulator(noise_model)
        qc_transpiled = self._get_transpiled_circuit(qc_measure, simulator)
        result = simulator.run(qc_transpiled, shots=self.shots).result()
        counts = result.get_counts()

        num_states = 2 ** self.num_qubits
        probs = np.zeros(num_states)

        for bitstring, count in counts.items():
            state_idx = int(bitstring, 2)
            probs[state_idx] = count / self.shots

        return probs
    
    def get_expectation_values(self, qc: QuantumCircuit, observables: List[str],
                              noise_model: Optional[NoiseModel] = None) -> np.ndarray:
        expectations = []
        simulator = self._get_simulator(noise_model)

        for observable in observables:
            qc_copy = qc.copy()

            for i, pauli in enumerate(observable):
                if pauli == 'X':
                    qc_copy.h(i)
                elif pauli == 'Y':
                    qc_copy.sdg(i)
                    qc_copy.h(i)

            qc_copy.measure_all()

            qc_transpiled = self._get_transpiled_circuit(qc_copy, simulator)
            result = simulator.run(qc_transpiled, shots=self.shots).result()
            counts = result.get_counts()

            expectation = 0.0
            for bitstring, count in counts.items():
                parity = sum([int(bit) for bit in bitstring]) % 2
                sign = 1 if parity == 0 else -1
                expectation += sign * count / self.shots

            expectations.append(expectation)

        return np.array(expectations)
    
    def get_correlation_terms(self, qc: QuantumCircuit, noise_model: Optional[NoiseModel] = None) -> np.ndarray:
        correlations = []
        simulator = self._get_simulator(noise_model)

        qc_copy = qc.copy()
        qc_copy.measure_all()

        qc_transpiled = self._get_transpiled_circuit(qc_copy, simulator)
        result = simulator.run(qc_transpiled, shots=self.shots).result()
        counts = result.get_counts()

        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                correlation = 0.0
                for bitstring, count in counts.items():
                    bit_i = int(bitstring[-(i+1)])
                    bit_j = int(bitstring[-(j+1)])
                    sign_i = 1 if bit_i == 0 else -1
                    sign_j = 1 if bit_j == 0 else -1
                    correlation += sign_i * sign_j * count / self.shots

                correlations.append(correlation)

        return np.array(correlations)
    
    def _process_single_sample(self, params: Tuple) -> Tuple[np.ndarray, np.ndarray, int, str, float, np.ndarray]:
        depth, noise_type, noise_strength, circuit_idx, circuit_type, feature_type, max_depth = params

        if circuit_type == 'random':
            qc = self.create_random_circuit(depth, seed=circuit_idx)
        else:
            qc = self.create_variational_circuit(depth)

        noise_model = self.create_noise_model(noise_type, noise_strength)

        if feature_type == 'expectation':
            observables = []
            for i in range(self.num_qubits):
                obs = ['I'] * self.num_qubits
                obs[i] = 'Z'
                observables.append(''.join(obs))
            x_ideal = self.get_expectation_values(qc, observables, noise_model=None)
            x_noisy = self.get_expectation_values(qc, observables, noise_model=noise_model)
        elif feature_type == 'bitstring':
            x_ideal = self.get_bitstring_probabilities(qc, noise_model=None)
            x_noisy = self.get_bitstring_probabilities(qc, noise_model=noise_model)
        elif feature_type == 'correlation':
            x_ideal = self.get_correlation_terms(qc, noise_model=None)
            x_noisy = self.get_correlation_terms(qc, noise_model=noise_model)

        noise_descriptor = np.array([
            depth / max_depth,
            noise_strength,
            {'depolarizing': 0, 'amplitude_damping': 1, 'readout': 2, 'mixed': 3}[noise_type] / 3.0
        ])

        return x_noisy, x_ideal, depth, noise_type, noise_strength, noise_descriptor

    def generate_dataset(self,
                        num_circuits: int,
                        depths: List[int],
                        noise_types: List[str],
                        noise_strengths: List[float],
                        circuit_type: str = 'random',
                        feature_type: str = 'bitstring',
                        n_jobs: int = -1) -> Dict:

        total_samples = num_circuits * len(depths) * len(noise_types) * len(noise_strengths)

        if feature_type == 'bitstring':
            feature_dim = 2 ** self.num_qubits
        elif feature_type == 'expectation':
            feature_dim = self.num_qubits
        elif feature_type == 'correlation':
            feature_dim = self.num_qubits * (self.num_qubits - 1) // 2

        dataset = {
            'x_noisy': np.zeros((total_samples, feature_dim)),
            'x_ideal': np.zeros((total_samples, feature_dim)),
            'noise_descriptors': np.zeros((total_samples, 3)),
            'circuit_depths': np.zeros(total_samples, dtype=int),
            'noise_types': [],
            'noise_strengths': np.zeros(total_samples)
        }

        params_list = []
        for depth in depths:
            for noise_type in noise_types:
                for noise_strength in noise_strengths:
                    for circuit_idx in range(num_circuits):
                        params_list.append((depth, noise_type, noise_strength, circuit_idx,
                                          circuit_type, feature_type, max(depths)))

        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs == 1:
            n_jobs = None

        print(f"Generating {total_samples} samples with {n_jobs if n_jobs else 1} worker(s)...")

        if n_jobs is None or n_jobs == 1:
            for idx, params in enumerate(params_list):
                result = self._process_single_sample(params)
                x_noisy, x_ideal, depth, noise_type, noise_strength, noise_descriptor = result

                dataset['x_noisy'][idx] = x_noisy
                dataset['x_ideal'][idx] = x_ideal
                dataset['circuit_depths'][idx] = depth
                dataset['noise_types'].append(noise_type)
                dataset['noise_strengths'][idx] = noise_strength
                dataset['noise_descriptors'][idx] = noise_descriptor

                if (idx + 1) % 10 == 0:
                    print(f"Generated {idx + 1}/{total_samples} samples...")
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(self._process_single_sample, params): idx
                          for idx, params in enumerate(params_list)}

                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    x_noisy, x_ideal, depth, noise_type, noise_strength, noise_descriptor = result

                    dataset['x_noisy'][idx] = x_noisy
                    dataset['x_ideal'][idx] = x_ideal
                    dataset['circuit_depths'][idx] = depth
                    dataset['noise_types'].append(noise_type)
                    dataset['noise_strengths'][idx] = noise_strength
                    dataset['noise_descriptors'][idx] = noise_descriptor

                    completed += 1
                    if completed % 10 == 0:
                        print(f"Generated {completed}/{total_samples} samples...")

        print(f"Dataset generation complete: {total_samples} total samples")
        print(f"Feature dimension: {dataset['x_noisy'].shape[1]}")

        return dataset
    
    def generate_dataset_batch(self,
                              num_circuits: int,
                              depths: List[int],
                              noise_types: List[str],
                              noise_strengths: List[float],
                              circuit_type: str = 'random',
                              feature_type: str = 'bitstring',
                              batch_size: int = 100,
                              n_jobs: int = -1) -> Dict:

        total_samples = num_circuits * len(depths) * len(noise_types) * len(noise_strengths)

        if feature_type == 'bitstring':
            feature_dim = 2 ** self.num_qubits
        elif feature_type == 'expectation':
            feature_dim = self.num_qubits
        elif feature_type == 'correlation':
            feature_dim = self.num_qubits * (self.num_qubits - 1) // 2

        dataset = {
            'x_noisy': np.zeros((total_samples, feature_dim)),
            'x_ideal': np.zeros((total_samples, feature_dim)),
            'noise_descriptors': np.zeros((total_samples, 3)),
            'circuit_depths': np.zeros(total_samples, dtype=int),
            'noise_types': [],
            'noise_strengths': np.zeros(total_samples)
        }

        params_list = []
        for depth in depths:
            for noise_type in noise_types:
                for noise_strength in noise_strengths:
                    for circuit_idx in range(num_circuits):
                        params_list.append((depth, noise_type, noise_strength, circuit_idx,
                                          circuit_type, feature_type, max(depths)))

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        print(f"Generating {total_samples} samples in batches of {batch_size} with {n_jobs} worker(s)...")

        for batch_start in range(0, len(params_list), batch_size):
            batch_end = min(batch_start + batch_size, len(params_list))
            batch_params = params_list[batch_start:batch_end]

            if n_jobs == 1:
                results = [self._process_single_sample(params) for params in batch_params]
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    results = list(executor.map(self._process_single_sample, batch_params))

            for local_idx, result in enumerate(results):
                global_idx = batch_start + local_idx
                x_noisy, x_ideal, depth, noise_type, noise_strength, noise_descriptor = result

                dataset['x_noisy'][global_idx] = x_noisy
                dataset['x_ideal'][global_idx] = x_ideal
                dataset['circuit_depths'][global_idx] = depth
                dataset['noise_types'].append(noise_type)
                dataset['noise_strengths'][global_idx] = noise_strength
                dataset['noise_descriptors'][global_idx] = noise_descriptor

            self._transpile_cache.clear()

            print(f"Generated {batch_end}/{total_samples} samples...")

        print(f"Dataset generation complete: {total_samples} total samples")
        print(f"Feature dimension: {dataset['x_noisy'].shape[1]}")

        return dataset

    def save_dataset(self, dataset: Dict, filename: str):
        np.savez(filename, **dataset)
        print(f"Dataset saved to {filename}")

    def load_dataset(self, filename: str) -> Dict:
        data = np.load(filename, allow_pickle=True)
        dataset = {key: data[key] for key in data.files}
        print(f"Dataset loaded from {filename}")
        return dataset


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("QEM Data Generator - Example")
    print("=" * 50)

    generator = QEMDataGenerator(num_qubits=3, shots=8192)

    dataset = generator.generate_dataset(
        num_circuits=5,
        depths=[2, 4, 6],
        noise_types=['depolarizing', 'amplitude_damping'],
        noise_strengths=[0.01, 0.05, 0.1],
        circuit_type='random',
        feature_type='bitstring'
    )

    generator.save_dataset(dataset, 'qem_dataset_example.npz')

    print("\nDataset summary:")
    print(f"Total samples: {len(dataset['x_noisy'])}")
    print(f"Input dimension: {dataset['x_noisy'].shape}")
    print(f"Output dimension: {dataset['x_ideal'].shape}")
    print(f"Noise descriptor dimension: {dataset['noise_descriptors'].shape}")
