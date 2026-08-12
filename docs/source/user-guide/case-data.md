# Case data

`pwrs` represents a network as a `MatpowerCase`. Its standard fields follow
the MATPOWER case struct, while numeric matrices are NumPy arrays or, for the
generalized matrices `A`, `N`, and `H`, SciPy sparse matrices.

## Built-in cases

The package exposes built-in MATPOWER and PGLib-OPF cases through `pwrs` and
the data subpackages:

```python
import pwrs as mp
from pwrs import case9

case_a = mp.case9()
case_b = case9()

# Data sets remain grouped under pwrs.data.
case_c = mp.data.matpower.case9()
pglib_case = mp.data.pglibopf.pglib_opf_case14_ieee()
```

The first three forms load the same case. MATPOWER case functions remain
available from the root package for compatibility, while `pwrs.data` groups
the `matpower` and `pglibopf` collections explicitly. Packaged cases use the
canonical JSON representation described below.

## Unified load and save API

Use `pwrs.load()` and `pwrs.save()` for case files. The format is normally
inferred from the path suffix:

```python
from pathlib import Path

import pwrs as mp

source = mp.case30()
path = mp.save(source, "case30.json")
loaded = mp.load(path)

assert isinstance(path, Path)
assert loaded.bus.shape == source.bus.shape
```

Both functions are also available from `pwrs.corex.io`. `read_case()` and
`write_case()` are aliases; format-specific functions such as `read_json()`
and `write_npz()` are available when an application intentionally requires a
particular representation.

The supported formats are:

| Format | Suffix | Intended use |
| --- | --- | --- |
| JSON | `.json` | Canonical, human-readable, reviewable case data |
| MAT-file | `.mat` | Exchange with MATLAB and MATPOWER |
| NumPy archive | `.npz` | Fast, pickle-free local loading |
| Excel workbook | `.xlsx` | Sheet-oriented inspection and editing |
| PYPOWER mapping | none | In-memory interoperability with dictionary-based code |

To use a path without a suffix, specify `format`; the appropriate suffix is
then appended:

```python
path = mp.save(mp.case9(), "case9-copy", format="npz")
case = mp.load("case9-copy", format="npz")

assert path.name == "case9-copy.npz"
```

Accepted file format names are `"json"`, `"mat"`, `"npz"`, `"excel"`, and
`"xlsx"`. An explicit format that conflicts with an existing suffix is
rejected instead of silently writing a differently encoded file. Legacy
`.xls` workbooks and MATLAB `.m` case functions are not part of the unified
file API.

## JSON

JSON is the canonical repository format. It is stable and readable, places
each dense matrix row on one line, and records array dtypes so that a
round-trip does not convert integer arrays to floating point. For example:

```python
mp.write_json(mp.case9(), "case9.json")
case = mp.read_json("case9.json")
```

Canonical files contain schema metadata and a `__dtypes__` table. Sparse
`A`, `N`, and `H` matrices use an explicit COO representation. Non-finite
numeric values are encoded as `"Infinity"`, `"-Infinity"`, and `"NaN"`,
keeping the file valid standards-compliant JSON while restoring the original
numeric values on load.

The reader also accepts older packaged JSON cases without schema or dtype
metadata. New files written by `pwrs` always use the canonical schema.

## MAT-files

MAT-files contain a MATLAB struct. The default variable name is `mpc`, matching
MATPOWER conventions:

```python
mp.save(mp.case30(), "case30.mat")
case = mp.load("case30.mat")
```

Use `variable_name` when the struct has another name:

```python
mp.save(mp.case30(), "network.mat", variable_name="network")
case = mp.load("network.mat", variable_name="network")
```

The name must be a valid MATLAB variable name. Loading fails clearly when the
requested variable is absent or is not a struct.

## NPZ

NPZ is intended for repeated local loads where text review is not required:

```python
mp.save(mp.case1354pegase(), "case1354pegase.npz")
case = mp.load("case1354pegase.npz")
```

The archive is uncompressed to minimize decoding overhead. It preserves array
dtypes and stores sparse matrices as validated CSC components. Loading always
uses `allow_pickle=False`; object arrays and archives with missing or unknown
schema metadata are rejected. This makes NPZ suitable for development caches,
but JSON should remain the checked-in source representation when human diff
and review matter.

## Excel

Excel represents each case field on a separate sheet. Standard MATPOWER
matrices have named columns, vector and string fields have a `value` column,
and sparse matrices use `row`, `column`, and `value` columns. A required
`_meta` sheet records the schema, dtype, shape, and scalar fields:

```python
mp.save(mp.case30(), "case30.xlsx")
case = mp.load("case30.xlsx")
```

Excel support requires `openpyxl`, included in the `all` dependency group:

```bash
uv sync --group all
```

The reader validates headers, declared shapes, numeric dtypes, sparse
coordinates, and sheet registration. Extra sheets are rejected because they
would otherwise look like case data but be silently ignored. Use the
`comments` case field rather than an unregistered notes sheet.

Legacy `.xls` files are not supported; save them as `.xlsx` first.

## PYPOWER dictionaries

`load()` accepts a PYPOWER-style mapping directly and returns an independent
`MatpowerCase`:

```python
ppc = {
    "version": "2",
    "baseMVA": 100.0,
    "bus": [[1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]],
    "gen": [[1, 0, 0, 10, -10, 1, 100, 1, 10, 0]],
    "branch": [],
}

case = mp.load(ppc)
ppc_copy = mp.to_pypower(case)
```

`from_pypower()` is the explicit conversion equivalent to `load(ppc)`, and
`to_pypower()` returns a deep, independent dictionary. Mutating either the
input mapping or converted output therefore does not mutate the case.

The `format` argument for a mapping may be omitted or set to `"pypower"` or
`"dict"`. `save()` always writes a file, so use `to_pypower()` when the desired
output is another in-memory mapping.

## Validation and preserved fields

All formats pass through one canonical schema. A case must contain
`baseMVA`, `bus`, `gen`, and `branch`. The supported optional fields are:

- dense matrices: `gencost`, `dcline`, `dclinecost`, `areas`, and `fparm`;
- dense or sparse matrices: `A`, `N`, and `H`;
- vectors: `l`, `u`, `Cw`, `z0`, `zl`, and `zu`;
- string lists: `gentype`, `genfuel`, `bus_name`, `branch_name`, and
  `comments`;
- scalars: `version`, `success`, `iterations`, `et`, and `f`.

Dense matrices are normalized to two dimensions, vectors to one dimension,
and inputs are copied. Deprecated `busname` and `branchname` fields are
normalized to `bus_name` and `branch_name`. Unknown fields, conflicting
aliases, invalid shapes, and invalid types are rejected rather than silently
dropped.

## Saving solved cases

Structured results from `runpf()`, `runopf()`, and related functions can be
saved directly as reusable cases:

```python
options = mp.mpoption("out.all", 0, "verbose", 0)
result = mp.runpf(mp.case30(), options)

mp.save(result, "case30-solved.npz")
solved_case = mp.load("case30-solved.npz")
```

The solved bus, generator, and branch matrices and standard result scalars
such as `success`, `iterations`, `et`, and `f` are preserved when present.
Runtime solver state such as internal ordering, model objects, multipliers,
and raw solver output is deliberately omitted because it is not reusable case
data and may contain arbitrary Python objects.

## Deprecated compatibility functions

`loadcase()` and `savecase()` remain available for MATPOWER compatibility but
emit `DeprecationWarning` for data IO. New code should use `pwrs.load()` and
`pwrs.save()`.

The legacy `savecase()` path is still required when generating a MATLAB `.m`
case function. The unified API intentionally handles data formats only and
does not execute or generate Python or MATLAB source files.
