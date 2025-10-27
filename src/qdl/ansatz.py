from __future__ import annotations

from typing import Sequence

import pennylane as qml # type: ignore


def apply_ansatz(
    weights: dict,
    wires: Sequence[int],
    measurement_wires: Sequence[int],
    use_trainable_post_rotations: bool,
    ansatz_type: str = "strongly_entangling",
) -> None:
    """Ansatz parametrik tingkat lanjut dengan opsi adaptif."""
    ansatz_type = ansatz_type.lower()
    entangling = weights["entangling"]
    ising = weights["ising"]
    local_rot = weights["local_rot"]

    num_layers = ising.shape[0]
    num_qubits = len(wires)

    if ansatz_type == "strongly_entangling":
        qml.StronglyEntanglingLayers(entangling, wires=wires)
        for layer_idx in range(num_layers):
            for wire_idx in range(num_qubits):
                src = wires[wire_idx]
                tgt = wires[(wire_idx + 1) % num_qubits]
                qml.IsingZZ(ising[layer_idx, wire_idx], wires=[src, tgt])
            for wire_idx in range(num_qubits):
                qml.RY(local_rot[layer_idx, wire_idx, 0], wires=wires[wire_idx])
                qml.RZ(local_rot[layer_idx, wire_idx, 1], wires=wires[wire_idx])
    elif ansatz_type == "qresnet":
        for layer_idx in range(num_layers):
            qml.StronglyEntanglingLayers(entangling[layer_idx : layer_idx + 1], wires=wires)
            for wire_idx in range(num_qubits):
                qml.RY(local_rot[layer_idx, wire_idx, 0], wires=wires[wire_idx])
            qml.broadcast(qml.CZ, wires=wires, pattern="ring")
            for wire_idx in range(num_qubits):
                qml.RZ(local_rot[layer_idx, wire_idx, 1], wires=wires[wire_idx])
            for wire_idx in range(num_qubits):
                src = wires[wire_idx]
                tgt = wires[(wire_idx + 1) % num_qubits]
                qml.CRX(ising[layer_idx, wire_idx], wires=[src, tgt])
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    if use_trainable_post_rotations and "post_rot" in weights:
        for idx, wire in enumerate(measurement_wires):
            qml.RY(weights["post_rot"][idx], wires=wire)