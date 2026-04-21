"""Polynomial library utilities for monomial enumeration and multiplication tables."""

from itertools import combinations_with_replacement
from math import comb
from typing import List

import torch


def compute_library_size(n_features: int, degree: int) -> int:
    """Number of monomials up to degree D in n features.

    = C(n_features + degree, degree)
    Example: (n=3, d=2) -> 10 terms: {1, x0, x1, x2, x0^2, x0x1, x0x2, x1^2, x1x2, x2^2}
    """
    return comb(n_features + degree, degree)


def build_library_structure(n_features: int, degree: int) -> dict:
    """Enumerate monomials and build multiplication table for recursive expansion.

    Term ordering: all degree-0 terms, then degree-1, ..., up to degree-D.
    Within each degree: combinations_with_replacement order.

    Example for n_features=2, degree=2:
        terms     = [(), (0,), (1,), (0,0), (0,1), (1,1)]
        names     = ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']
        n_terms   = 6
        bias_index = 0  (the constant term)
        linear_indices = [1, 2]  (indices of h and u)

    mult_table[t, f] = index of the monomial (term_t * x_f), or -1 if the
        product would exceed the maximum degree. Guaranteed unique targets:
        for fixed f, each source term maps to a distinct target (sorted
        multisets are unique), so accumulation via indexing is safe.

    Returns:
        terms: list of tuples (sorted feature index multisets)
        mult_table: (n_terms, n_features) long tensor
        linear_indices: (n_features,) long tensor
        bias_index: int
        n_terms: int
    """
    terms = [()]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            terms.append(combo)

    term_to_idx = {term: idx for idx, term in enumerate(terms)}
    n_terms = len(terms)

    mult_table = torch.full((n_terms, n_features), -1, dtype=torch.long)
    for t_idx, term in enumerate(terms):
        if len(term) < degree:
            for f in range(n_features):
                product = tuple(sorted(term + (f,)))
                if product in term_to_idx:
                    mult_table[t_idx, f] = term_to_idx[product]

    bias_index = term_to_idx[()]
    linear_indices = torch.tensor(
        [term_to_idx[(f,)] for f in range(n_features)], dtype=torch.long
    )

    return {
        'terms': terms,
        'mult_table': mult_table,
        'linear_indices': linear_indices,
        'bias_index': bias_index,
        'n_terms': n_terms,
    }


def get_library_feature_names(feature_names: List[str], degree: int) -> List[str]:
    """Generate human-readable monomial names matching build_library_structure order.

    Example: ['h', 'u'], degree=2 -> ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']
    """
    names = ['1']
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(len(feature_names)), d):
            counts = {}
            for idx in combo:
                counts[idx] = counts.get(idx, 0) + 1
            parts = []
            for idx, cnt in sorted(counts.items()):
                name = feature_names[idx]
                parts.append(f"{name}^{cnt}" if cnt > 1 else name)
            names.append('*'.join(parts))
    return names


def get_polynomial_degree_from_term(term: str) -> int:
    """Extract total degree from term string.

    '1'->0, 'x'->1, 'x^2'->2, 'x*y'->2, 'x^2*y'->3
    """
    if term == '1':
        return 0
    total = 0
    for part in term.split('*'):
        if '^' in part:
            total += int(part.split('^')[1])
        else:
            total += 1
    return total
