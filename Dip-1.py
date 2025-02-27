from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((2, 2))
    print("Original Pixel Values:\n", np.array(img.getdata()).reshape(2, 2))
    return np.array(img.getdata()).reshape(2, 2) / 255.0

def quantum_encode(image_data):
    flattened = image_data.flatten()
    norm = np.linalg.norm(flattened)
    if norm < 1e-10:  # Handle black/white images
        flattened = np.ones_like(flattened)
        norm = np.linalg.norm(flattened)
    normalized = flattened / norm
    
    print(f"Encoded State: {normalized}")
    qc = QuantumCircuit(2)
    qc.initialize(normalized, [0,1])
    return qc

def quantum_processing(qc):
    qc.barrier()
    qc.cx(0,1)
    qc.cx(1,0)
    qc.save_statevector()
    return qc

def measure_circuit(qc):
    simulator = AerSimulator(method='statevector')
    job = simulator.run(qc)
    result = job.result()
    statevector = result.data()['statevector']
    print(f"Final State Vector: {statevector}")
    return statevector

# Main workflow with debugging
image_matrix = preprocess_image(r"C:\Users\harsh\Desktop\EnhanceAfter.jpg")
quantum_circuit = quantum_encode(image_matrix)
processed_circuit = quantum_processing(quantum_circuit)
statevector = measure_circuit(processed_circuit)

probabilities = np.abs(statevector)**2
print(f"Raw Probabilities: {probabilities}")

# Enhanced output processing
output_image = (probabilities * 255).reshape(2,2).astype(np.uint8)
print("Processed Image Matrix:\n", output_image)

# Improved display with scaling
img = Image.fromarray(output_image)
img = img.resize((200, 200), resample=Image.NEAREST)  # Upscale for visibility
img.show()
