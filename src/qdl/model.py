from __future__ import annotations

import pennylane as qml # type: ignore
import torch # type: ignore
from torch import nn # type: ignore
from .ansatz import apply_ansatz
from .config import ModelConfig
from .feature_maps import apply_feature_map
class QuantumDeepLearningModel(nn.Module):
    """Model QDL hibrida (quantum embedding + classical head)."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wires = list(range(cfg.n_qubits))
        self.measurement_wires = cfg.measurement_wires
        self.dev = qml.device(cfg.device, wires=cfg.n_qubits, shots=cfg.shots)
        self.quantum_layer = self._build_quantum_layer()
        self.classical_head = nn.Sequential(
            nn.LayerNorm(len(self.measurement_wires)),
            nn.Linear(len(self.measurement_wires), cfg.classical_head_width),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(cfg.classical_head_width, cfg.output_dim),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _weight_shapes(self) -> dict:
        shapes = {
            "entangling": (self.cfg.circuit_layers, self.cfg.n_qubits, 3),
            "ising": (self.cfg.circuit_layers, self.cfg.n_qubits),
            "local_rot": (self.cfg.circuit_layers, self.cfg.n_qubits, 2),
        }
        if self.cfg.use_trainable_post_rotations:
            shapes["post_rot"] = (len(self.measurement_wires),)
        return shapes

    def _build_qnode(self):
        feature_strategy = self.cfg.feature_map
        ansatz_kind = self.cfg.ansatz
        wires = self.wires
        measurement_wires = self.measurement_wires
        use_post_rot = self.cfg.use_trainable_post_rotations

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, **weights):
            apply_feature_map(inputs, wires=wires, strategy=feature_strategy)
            apply_ansatz(
                weights,
                wires=wires,
                measurement_wires=measurement_wires,
                use_trainable_post_rotations=use_post_rot,
                ansatz_type=ansatz_kind,
            )
            return [qml.expval(qml.PauliZ(w)) for w in measurement_wires]

        return circuit

    def _build_quantum_layer(self):
        qnode = self._build_qnode()
        weight_shapes = self._weight_shapes()
        return qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        quantum_features = self.quantum_layer(x)
        logits = self.classical_head(quantum_features)
        return self.log_softmax(logits)
    def freeze_quantum_layer(self) -> None:
        """Freeze semua parameter quantum layer untuk transfer learning."""
        for param in self.quantum_layer.parameters():
            param.requires_grad = False
        console.log("[cyan]Quantum layer frozen for transfer learning.[/cyan]") # type: ignore
    def unfreeze_quantum_layer(self) -> None:
        """Buka freeze quantum layer jika ingin melanjutkan training penuh."""
        for param in self.quantum_layer.parameters():
            param.requires_grad = True
        console.log("[cyan]Quantum layer unfrozen.[/cyan]") # type: ignore
