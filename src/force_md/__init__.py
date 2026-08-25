"""Force-conditioned protein MD ensemble model.

Phase 1 implements the hierarchical local-physics model: a three-level
atom / residue / backbone-frame graph, an e3nn SE(3)-equivariant encoder at
``l_max=2``, and force / torque / energy / uncertainty heads. Phase 2 loads
these modules unchanged and adds the temporal stochastic transition.
"""

__version__ = "0.1.0.dev0"
