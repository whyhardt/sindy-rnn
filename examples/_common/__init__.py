"""Shared, model-agnostic evaluation and plotting helpers for the
lorenz/cylinder/sst example benchmarks.

Everything here operates on plain arrays (predictions, ground truth) so it
can score/plot results from any method (sindy-rnn, SINDy-SHRED, STLSQ, ...)
identically, per the shared-evaluation-protocol rule in CLAUDE.md.
"""
