import torch # type: ignore

from qdl.config import ModelConfig
from qdl.model import QuantumDeepLearningModel


def test_forward_shape():
    cfg = ModelConfig(
        n_qubits=4,
        circuit_layers=2,
        measurement_wires=[0, 1],
        output_dim=2,
    )
    model = QuantumDeepLearningModel(cfg)
    dummy = torch.randn(8, cfg.n_qubits)
    out = model(dummy)
    assert out.shape == (8, cfg.output_dim)