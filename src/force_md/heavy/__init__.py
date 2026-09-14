"""Heavy-atom extension (H0-H2) for the Phase 1.6 transition probe.

Kept in its own package so that nothing in ``force_md.transition`` changes
behaviour: every existing code path produces the same output with this package
installed as without it.
"""
