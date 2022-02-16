import numpy as np
from sklearn import decomposition

# NMF
from sklearn._config import config_context
from sklearn.decomposition._nmf \
    import _beta_divergence, \
    _check_string_param, \
    _check_init, \
    _initialize_nmf, \
    _compute_regularization
from sklearn.utils.validation \
    import check_array, \
    check_non_negative, \
    check_random_state
from scipy import sparse
# from scipy.sparse.base import issparse
# import warnings
import numbers

# from sklearn.decomposition._nmf import _initialize_nmf as init_NMF

import cupy as cp


def cupy_safe_sparse_dot(a, b, *, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = cp.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sparse.issparse(a) and sparse.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret


def _cupy_update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle,
                                    random_state):
    n_components = Ht.shape[1]

    HHt = cp.dot(Ht.T, Ht)
    XHt = cupy_safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.:
        # adds l2_reg only on the diagonal
        HHt.flat[::n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    # return _update_cdnmf_fast(W, HHt, XHt, permutation)


def _cupy_fit_coordinate_descent(X, W, H, tol=1e-4, max_iter=200, l1_reg_W=0,
                                 l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                                 update_H=True, verbose=0, shuffle=False,
                                 random_state=None):
    Ht = check_array(H.T, order='C')
    X = check_array(X, accept_sparse='csr')

    rng = check_random_state(random_state)

    Ht = cp.asarray(Ht)
    W = cp.asarray(W)
    X = cp.asarray(X)

    for n_iter in range(1, max_iter + 1):
        violation = 0.

        # Update W
        violation += _cupy_update_coordinate_descent(X, W, Ht, l1_reg_W,
                                                     l2_reg_W, shuffle, rng)
        # Update H
        if update_H:
            # violation += _update_coordinate_descent(X.T, Ht, W, l1_reg_H,
            #                                         l2_reg_H, shuffle, rng)
            pass

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def cupy_non_negative_factorization(X, W=None, H=None, n_components=None, *,
                                    init='warn', update_H=True, solver='cd',
                                    beta_loss='frobenius', tol=1e-4,
                                    max_iter=200, alpha=0., l1_ratio=0.,
                                    regularization=None, random_state=None,
                                    verbose=0, shuffle=False):
    X = check_array(X, accept_sparse=('csr', 'csc'),
                    dtype=[np.float64, np.float32])
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if X.min() == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, numbers.Integral) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, numbers.Integral) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
        if H.dtype != X.dtype or W.dtype != X.dtype:
            raise TypeError("H and W should have the same dtype as X. Got "
                            "H.dtype = {} and W.dtype = {}."
                            .format(H.dtype, W.dtype))
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        if H.dtype != X.dtype:
            raise TypeError("H should have the same dtype as X. Got H.dtype = "
                            "{}.".format(H.dtype))
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = np.sqrt(X.mean() / n_components)
            W = np.full((n_samples, n_components), avg, dtype=X.dtype)
        else:
            W = np.zeros((n_samples, n_components), dtype=X.dtype)
    else:
        W, H = _initialize_nmf(X, n_components, init=init,
                               random_state=random_state)

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
        alpha, l1_ratio, regularization)

    if solver == 'cd':
        W, H, n_iter = _cupy_fit_coordinate_descent(X, W, H, tol, max_iter,
                                                    l1_reg_W, l1_reg_H,
                                                    l2_reg_W, l2_reg_H,
                                                    update_H=update_H,
                                                    verbose=verbose,
                                                    shuffle=shuffle,
                                                    random_state=random_state)
    elif solver == 'mu':
        print('未実装 (solver == \'mu\')')

    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        # warnings.warn("Maximum number of iterations %d reached."
        #     "Increase it to"
        #     " improve convergence." % max_iter, ConvergenceWarning)
        pass

    return W, H, n_iter


class cuNMF(decomposition.NMF):
    def fit_transform(self, X, y=None, W=None, H=None):
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                dtype=[np.float64, np.float32])
        with config_context(assume_finite=True):
            W, H, n_iter_ = cupy_non_negative_factorization(
                X=X, W=W, H=H, n_components=self.n_components, init=self.init,
                update_H=True, solver=self.solver, beta_loss=self.beta_loss,
                tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
                l1_ratio=self.l1_ratio, regularization=self.regularization,
                random_state=self.random_state, verbose=self.verbose,
                shuffle=self.shuffle)

        self.reconstruction_err_ = _beta_divergence(X, W, H, self.beta_loss,
                                                    square_root=True)
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_
        return W
        # return super().fit_transform(X, y, W, H)


X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
model = cuNMF(init='nndsvd')
model.fit_transform(X)

# decomposition.NMF()
