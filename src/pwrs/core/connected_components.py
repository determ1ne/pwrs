# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def connected_components(C, groups=None, unvisited=None, nargout=None):
    """Find connected components of a node-branch incidence matrix.

    Mirrors MATPOWER's ``connected_components`` helper. It performs a graph
    traversal on the incidence matrix ``C`` to identify connected groups of
    buses and, when requested, isolated nodes.

    Parameters
    ----------
    C : sparse matrix or array_like
        Node-branch incidence matrix.
    groups : list, optional
        Existing group accumulator used by the recursive implementation.
    unvisited : array_like, optional
        Remaining one-based node indices to explore.
    nargout : int, optional
        MATLAB compatibility flag controlling whether isolated nodes are also
        returned.

    Returns
    -------
    list or tuple
        Connected groups, and optionally the isolated node indices.
    """
    C = sparse.csc_matrix(C)
    nn = C.shape[1]
    Ct = C.transpose().tocsc()
    visited = np.zeros(nn, dtype=int)
    if groups is None:
        groups = []
        unvisited = np.arange(1, nn + 1, dtype=int)
        isolated = np.flatnonzero(np.asarray(np.abs(C).sum(axis=0)).reshape(-1) == 0) + 1
        if isolated.size:
            unvisited = unvisited[~np.isin(unvisited, isolated)]
    else:
        isolated = None
        groups = [np.asarray(g).reshape(-1, 1) for g in groups]
        unvisited = np.asarray(unvisited, dtype=int).reshape(-1)

    cn = int(unvisited[0])
    visited[cn - 1] = 1
    qs = np.zeros(nn, dtype=int)
    f = 0
    N = 0
    qs[(f + N) % nn] = cn
    N += 1

    while N:
        cn = qs[f]
        N -= 1
        f = (f + 1) % nn
        mask = C[:, cn - 1] != 0
        jj = Ct[:, np.asarray(mask.toarray()).reshape(-1)].nonzero()[0] + 1
        cnn = jj[visited[jj - 1] == 0]
        for k in range(len(cnn)):
            node = int(cnn[k])
            if visited[node - 1] == 0:
                visited[node - 1] = 1
                N += 1
                qs[(f + N - 1) % nn] = node

    group = np.flatnonzero(visited) + 1
    groups.append(group.reshape(-1, 1))

    v = np.ones(nn, dtype=int)
    v[unvisited - 1] = 0
    v[group - 1] = 1
    unvisited = np.flatnonzero(v == 0) + 1

    if unvisited.size == 0:
        lengths = np.array([len(g) for g in groups], dtype=int)
        order = np.argsort(lengths)[::-1]
        groups = [groups[i] for i in order]
    else:
        groups, unvisited = connected_components(C, groups, unvisited, nargout=2)

    if isolated is not None:
        unvisited = isolated.reshape(-1, 1)

    if nargout == 1 or nargout is None:
        return groups
    return groups, unvisited
