"""In-repo benchmark harness: v3 candidate vs legacy reference.

The harness is a regression gate, not a performance goal. Every benchmark is
gated on solution parity against the legacy implementation before any timing
is recorded; a faster wrong result is a failed benchmark.
"""
