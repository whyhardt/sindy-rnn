"""sindy-rnn: Polynomial RNN for Sparse Nonlinear Dynamics Discovery."""

from .model import PolynomialRNN, EnsembleRNNModule, EnsemblePolynomialLayer, DecomposedPolynomialLayer, EnsembleLinear
from .training import fit
from .polynomial_library import (
    build_library_structure,
    compute_library_size,
    get_library_feature_names,
    get_polynomial_degree_from_term,
)
from .equations import get_coefficients, get_equations, get_continuous_equations
from .pruning import ensemble_prune, threshold_prune, agreement_test
from .rollout import RolloutSINDyRNN, fit_rollout, refit_rollout

__all__ = [
    'PolynomialRNN',
    'fit',
    'EnsembleRNNModule',
    'EnsemblePolynomialLayer',
    'DecomposedPolynomialLayer',
    'EnsembleLinear',
    'build_library_structure',
    'compute_library_size',
    'get_library_feature_names',
    'get_polynomial_degree_from_term',
    'get_coefficients',
    'get_equations',
    'ensemble_prune',
    'threshold_prune',
    'agreement_test',
    'RolloutSINDyRNN',
    'fit_rollout',
    'refit_rollout',
]
