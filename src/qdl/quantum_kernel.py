import pennylane as qml # type: ignore
from pennylane import numpy as np # type: ignore
import torch # type: ignore
from torch import nn # type: ignore

# Quantum device
dev_kernel = qml.device("lightning.qubit", wires=4)

def vq_kernel_circuit(x1, x2, weights):
    # Simple feature map
    qml.AngleEmbedding(x1, wires=range(4))
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    qml.AngleEmbedding(x2, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# TorchLayer wrapper untuk integrasi ke model
class VariationalQuantumKernel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.qnode = qml.QNode(vq_kernel_circuit, dev_kernel, interface="torch")

    def forward(self, x1, x2):
        return self.qnode(x1, x2, self.weights)
