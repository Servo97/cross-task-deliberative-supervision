"""Training-free KV compaction (Attention Matching) on the official FrameSamp+Modul VLA.

This namespace owns the H10 build: the reviewed AM patch of the scanned
``MemoryAttention``, disjoint action-query banks, per-layer artifact fitting,
the E0 velocity-matching evaluator, and the E0 staging/entry/launch plumbing.

Everything under ``robomme_integration/training/`` and the official policy
source tree is imported read-only; nothing here edits the Attention-Matching
kernel, the sealed-artifact pipeline, or the released model source.
"""
