"""Quick sanity check: 1 seed × 3 methods × 2 benchmarks.

Runs cylinder and SST benchmarks with NUM_SEEDS=1 to verify
all methods work correctly before launching full 5-seed studies.
"""
import sys
import os
import time

# Patch imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'sindy-shred'))

import math
import numpy as np
np.math = math

print("=" * 70)
print("SANITY CHECK: Cylinder Flow (1 seed)")
print("=" * 70)

# Monkey-patch NUM_SEEDS before importing
import examples.cylinder_benchmark as cyl
cyl.NUM_SEEDS = 1
t0 = time.time()
cyl.main()
print(f"\nCylinder sanity check completed in {time.time()-t0:.0f}s")

print("\n\n")
print("=" * 70)
print("SANITY CHECK: SST (1 seed)")
print("=" * 70)

import examples.sst_benchmark as sst
sst.NUM_SEEDS = 1
t0 = time.time()
sst.main()
print(f"\nSST sanity check completed in {time.time()-t0:.0f}s")

print("\n\nSanity checks complete. Review results above before launching full studies.")
