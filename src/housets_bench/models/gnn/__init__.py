"""Graph neural network baselines."""
from .gnn_forecaster import (  # noqa: F401  (registers all four GNN models)
    GCNTCNForecaster,
    GraphWaveNetForecaster,
    STGCNForecaster,
    STSGCNForecaster,
    STSGCNet,
)
