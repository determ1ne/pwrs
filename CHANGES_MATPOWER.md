- varargin/varargout functions have been split into separate functions.

    This includes:

    - ext2int (ext2int_mpc, ext2int_old)
    - loadcase (loadcase_embedded)

- Structs are typed into dataclasses:

    - mpc: the case file
    - mpoption: the option struct

- Several cases are backported from MATPOWER 8.1.

    - case1197
    - case11kundur
    - case145
    - case18
    - case300
    - case533mt_hi
    - case533mt_lo
    - case59
    - case_ieee30

- Functions not supported by pwrs:
    
    - Root package:
        - `Contents`: stub
        - `caseformat`: stub
        - `cdf2mpc`: not planned
        - `d2AIbr_dV2`: WIP
        - `d2ASbr_dV2`: WIP
        - `define_constants`: not applicable for Python code
        - `fmincopf`: fmincon is only available in MATLAB
        - `have_feature_*`: not applicable for pwrs
        - `miqps_matpower`: deprecated code, not planned
        - `mpoption_info_*`: not planned, strictly typed into dataclasses
        - `poly2pwl`: not planned
        - `psse*`: not planned
        - `runduopf`: not planned
        - `runopf_w_res`: not planned
        - `runuopf`: not planned
        - `toggle_iflims`: WIP, low priority
        - `toggle_reserves`: WIP, low priority
        - `toggle_softlims`: WIP, low priority
        - `uopf`: WIP, low priority
    - MIPS/PIPS:
        - `Contents`: stub
        - `have_feature_*`: not planned
    - MOST/POST:
        - `*`: WIP, low priority

- Functions has different calling conventions:

    - all:
        - The callee may modify the parameters, in-place.
    - `d2Sbr_dV2`: the connection matrix will always be dense

- Solvers supported in pwrs:

    - For mplinsolve:
        - Dense solver:
            - LAPACK _gesv: from numpy.linalg.solve
        - Sparse solver:
            - SuperLU: from scipy.sparse.linalg.splu / scipy.sparse.linalg.spsolve
            - KLU: from nbklu
    