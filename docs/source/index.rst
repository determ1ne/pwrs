pwrs
====

``pwrs`` is a typed Python library for power-system analysis and optimization.
It combines a MATPOWER-compatible calculation path with Python-native,
PowerModels-inspired formulations, providing power flow, continuation power
flow, and optimal power flow through a consistent Python API.

The MATPOWER-derived implementation preserves familiar case data and numerical
behavior, while the Python-native modeling stack supports multiple formulations
and external optimization solvers through Pyomo and CVXPY. Typed configuration
objects, structured results, and bundled MATPOWER and PGLib-OPF data make these
capabilities available without requiring MATLAB or Julia at runtime.

Start with the installation and a minimal power-flow example, then use the
workflow guides for more detailed operations. The API Reference is generated
from the Python source tree.


.. toctree::
   :maxdepth: 2
   :caption: User documentation

   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: Power System Internals

   internals/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
