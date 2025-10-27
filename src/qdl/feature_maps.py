from __future__ import annotations
from typing import Sequence
import pennylane as qml # type: ignore

# ✅ Coba import broadcast dengan fallback versi lama
try:
    # Pennylane >= 0.43
    from pennylane.ops.functions import broadcast as qml_broadcast # type: ignore
except ImportError:
    try:
       
        from pennylane.ops.op_math import broadcast as qml_broadcast # type: ignore
    except ImportError:
        
        qml_broadcast = None


def apply_feature_map(inputs, wires: Sequence[int], strategy: str = "hybrid") -> None:
    """Map fitur klasik ke state kuantum sesuai strategi."""
    strategy = strategy.lower()
    if strategy == "angle":
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
    elif strategy == "amplitude":
        qml.AmplitudeEmbedding(inputs, wires=wires, pad_with=0.0, normalize=True)
    elif strategy == "hybrid":
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        qml.AngleEmbedding(qml.math.sin(inputs), wires=wires, rotation="Z")
        if qml_broadcast:
            qml_broadcast(qml.CNOT, wires=wires, pattern="ring")
        else:
            for i in range(len(wires)):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
    elif strategy == "iqp":
        qml.IQPEmbedding(features=inputs, wires=wires)
    elif strategy == "fourier":
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        qml.AngleEmbedding(qml.math.sin(2.0 * inputs), wires=wires, rotation="Z")
        qml.AngleEmbedding(qml.math.cos(inputs), wires=wires, rotation="X")
    else:
        raise ValueError(f"Unsupported feature_map strategy: {strategy}")
