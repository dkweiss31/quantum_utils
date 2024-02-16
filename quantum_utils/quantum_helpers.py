import h5py
import numpy as np
import pathos
from typing import List
from qutip import tensor, Qobj, basis, qeye, to_kraus, sigmax, sigmay, sigmaz


def id_wrap_ops(op: Qobj, idx: int, truncated_dims: list) -> Qobj:
    """
    identity wrap the operator op which has index idx in a system
    where each subsystem has Hilbert dim as specified by truncated_dims
    Parameters
    ----------
    op: Qobj
        single subsystem operator
    idx: int
        position of the subsystem
    truncated_dims: list
        Hilbert space dimension of the subsystems

    Returns
    -------
    Qobj
    """
    assert op.dims[0][0] == truncated_dims[idx]
    id_list = [qeye(dim) for dim in truncated_dims]
    id_list[idx] = op
    return tensor(*id_list)


def calc_fidel_chi(chi_real, chi_ideal):
    return (4 * np.trace(chi_real @ chi_ideal) + np.trace(chi_real)) / 5


def construct_basis_states_list(
    Fock_states_spec: List[tuple], truncated_dims: List[int]
) -> List[Qobj]:
    """
    given Fock state specifications, return corresponding kets
    Parameters
    ----------
    Fock_states_spec: list[tuple]
        Fock state specifications. Ex: [(0, 1, 0), (1, 0, 0),] for a system with three subsystems,
        and the states requested are the excited state of the second subsystem and first subsyetm, respectively
    truncated_dims: list[int]
        Hilbert space dimension of the subsystems

    Returns
    -------
    list[Qobj]
    """
    if type(truncated_dims) == int:
        return [
            basis(truncated_dims, Fock_states_spec),
        ]
    basis_states = []
    for state_spec in Fock_states_spec:
        basis_list = [
            basis(truncated_dims[i], state_spec[i]) for i in range(len(truncated_dims))
        ]
        basis_states.append(tensor(*basis_list))
    return basis_states


def project_U(
    U: Qobj,
    Fock_states_spec: list = None,
    truncated_dims: list = None,
    basis_states: list = None,
):
    if basis_states is None:
        dim_new_U = len(Fock_states_spec)
        basis_states = np.zeros((dim_new_U, np.prod(truncated_dims)), dtype=complex)
        # converting to numpy array - this is a hack!
        for i, state_spec in enumerate(Fock_states_spec):
            basis_list = [
                basis(truncated_dims[j], state_spec[j])
                for j in range(len(truncated_dims))
            ]
            basis_states[i, :] = tensor(*basis_list).data.toarray()[:, 0]
        new_U = np.conjugate(basis_states) @ U.data.toarray() @ basis_states.T
        return Qobj(new_U)
    else:
        # assume basis_states is a list of kets
        dim_new_U = len(basis_states)
        new_U = np.zeros((dim_new_U, dim_new_U), dtype=complex)
        for i, basis_state_0 in enumerate(basis_states):
            for j, basis_state_1 in enumerate(basis_states):
                new_U[i, j] = (basis_state_0.dag() * U * basis_state_1).data.toarray()[
                    0, 0
                ]
        return Qobj(new_U)


def truncate_superoperator(superop, keep_idxs):
    """
    Parameters
    ----------
    superop
        superoperator to truncate. We consider the situation where certain
        states are relevant for predicting time evolution, however the gate under consideration
        does not care about the time evolution of those states themselves (with population
        e.g. beginning in that state.)
    keep_idxs
        indices of the states to keep
    Returns
    -------
        truncated superoperator

    """
    keep_dim = len(keep_idxs)
    total_dim = superop
    trunc_dim = total_dim - keep_dim
    truncated_mat = np.zeros((keep_dim**2, keep_dim**2), dtype=complex)
    total_dim = keep_dim + trunc_dim
    locs = [
        total_dim * keep_idx_i + keep_idx_j
        for keep_idx_i in keep_idxs
        for keep_idx_j in keep_idxs
    ]
    for i, loc_i in enumerate(locs):
        for j, loc_j in enumerate(locs):
            truncated_mat[i, j] = superop.data.toarray()[loc_i, loc_j]
    return Qobj(
        truncated_mat,
        type="super",
        dims=[[[keep_dim], [keep_dim]], [[keep_dim], [keep_dim]]],
    )


def my_to_chi(q_oper):
    """
    q_oper
        superoperator to transform into chi matrix
    """
    pauli_ops_oneq = [qeye(2) / 2, sigmax() / 2, sigmay() / 2, sigmaz() / 2]
    pauli_ops = [
        Qobj(tensor(pauli_op1, pauli_op2))
        for pauli_op1 in pauli_ops_oneq
        for pauli_op2 in pauli_ops_oneq
    ]
    kraus_ops = to_kraus(q_oper)
    e_ij_coeffs = np.array(
        [
            [np.trace(kraus_op.dag() * pauli_op) for pauli_op in pauli_ops]
            for kraus_op in kraus_ops
        ]
    )
    return np.conjugate(e_ij_coeffs).T @ e_ij_coeffs


def to_list_qobj(list_of_lists):
    """helper method for converting list of lists to list of qobjs"""
    return [Qobj(np.array(elem, dtype=complex)) for elem in list_of_lists]


def from_list_qobj(list_of_qobjs):
    """helper method for converting list of qobjs into numpy array for saving"""
    return np.stack([qobj_elem.data.toarray() for qobj_elem in list_of_qobjs])
