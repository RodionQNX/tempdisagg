import numpy as np
import pandas as pd


def mode_of_series(x):
    """Identify the type of series.

    Parameters
    ----------
    x : array-like or pandas object

    Returns
    -------
    str
        "ts" for pandas Series/DataFrame with a DatetimeIndex,
        "numeric" for numeric numpy arrays or lists.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return "ts"
    if isinstance(x, (np.ndarray, list, tuple)):
        return "numeric"
    raise TypeError("Series must be pandas or numeric")


def calc_c(n_l, conversion, fr, n_bc=0, n_fc=0):
    if n_bc < 0 or n_fc < 0:
        raise ValueError("n.bc and n.fc must be >= 0")
    if conversion == "sum":
        weights = np.ones(fr)
    elif conversion == "average":
        weights = np.ones(fr) / fr
    elif conversion == "first":
        weights = np.concatenate(([1], np.zeros(fr - 1)))
    elif conversion == "last":
        weights = np.concatenate((np.zeros(fr - 1), [1]))
    else:
        raise ValueError("Wrong type of conversion")
    C = np.kron(np.eye(n_l), weights.reshape(1, -1))
    if n_fc > 0:
        C = np.hstack([C, np.zeros((n_l, n_fc))])
    if n_bc > 0:
        C = np.hstack([np.zeros((n_l, n_bc)), C])
    return C


def calc_power_matrix(n):
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :])


def calc_r(rho, pm):
    return rho ** pm


def calc_q(rho, pm):
    return (1 / (1 - rho ** 2)) * calc_r(rho, pm)


def calc_q_lit(X, rho=0):
    n = X.shape[0]
    H = np.eye(n)
    D = np.eye(n)
    D[np.arange(1, n), np.arange(n-1)] = -1
    H[np.arange(1, n), np.arange(n-1)] = -rho
    inv = np.linalg.inv(D.T @ H.T @ H @ D)
    return inv


def calc_gls(y, X, vcov, logl=True, stats=True):
    m, n = X.shape
    if m <= n:
        raise ValueError("not enough degrees of freedom")

    b = y.reshape(-1, 1)
    B = np.linalg.cholesky(vcov).T

    qr_X = np.linalg.qr(X)
    Q = qr_X[0]
    R = qr_X[1]

    c_vec = Q.T @ b
    c1 = c_vec[:n]
    c2 = c_vec[n:]

    C_mat = Q.T @ B
    C1 = C_mat[:n]
    C2 = C_mat[n:]

    ftC2 = C2.T[::-1, ::-1]
    rq = np.linalg.qr(ftC2)
    PP = rq[0]
    SS = rq[1]
    P = PP[::-1, ::-1]
    S = SS[::-1, ::-1].T
    P1 = P[:, :n]
    P2 = P[:, n:]

    u2 = np.linalg.solve(S, c2)
    v = P2 @ u2
    x = np.linalg.solve(R, c1 - C1 @ v)

    z = {}
    z['coefficients'] = x.flatten()
    z['rss'] = float(u2.T @ u2)

    if logl:
        z['s_2'] = z['rss'] / m
        u_l = y.reshape(-1, 1) - X @ z['coefficients']
        z['logl'] = float(
            -m/2 - m*np.log(2*np.pi)/2 - m*np.log(z['s_2'])/2 - np.log(np.linalg.det(vcov))/2
        )
    if stats:
        z['s_2_gls'] = z['rss'] / (m - n)
        Lt = C1 @ P1
        R_inv = np.linalg.solve(R, np.eye(n))
        C_cov = R_inv @ Lt @ Lt.T @ R_inv.T
        z['se'] = np.sqrt(np.diag(z['s_2_gls'] * C_cov))
        vcov_inv = np.linalg.inv(vcov)
        e = np.ones((m, 1))
        y_bar = float(e.T @ vcov_inv @ y.reshape(-1, 1) / (e.T @ vcov_inv @ e))
        z['tss'] = float((y - y_bar).reshape(1, -1) @ vcov_inv @ (y - y_bar).reshape(-1, 1))
        z['rank'] = n
        z['df'] = m - n
        z['r.squared'] = 1 - z['rss'] / z['tss']
        z['adj.r.squared'] = 1 - (z['rss'] * (m - 1)) / (z['tss'] * (m - n))
        z['aic'] = np.log(z['rss'] / m) + 2 * (n / m)
        z['bic'] = np.log(z['rss'] / m) + np.log(m) * (n / m)
        z['vcov_inv'] = vcov_inv
    return z


def calc_dyn_adj(X, rho):
    n = X.shape[0]
    diag_rho = np.eye(n)
    diag_rho[1:, :] += (np.eye(n) * -rho)[:-1, :]
    rhs = np.hstack([X, np.r_[rho, np.zeros(n - 1)].reshape(-1, 1)])
    return np.linalg.solve(diag_rho, rhs)


def sub_regression_based(y_l, X, lf=None, lf_end=None, hf=None,
                         n_bc=None, n_fc=None, conversion="sum",
                         method="chow-lin-maxlog", fr=4,
                         truncated_rho=0, fixed_rho=0.5,
                         tol=1e-16, lower=-0.999, upper=0.999):
    n_l = len(y_l)
    n, m = X.shape

    if hf is not None:
        C = calc_clfhf(lf, hf, conversion, lf_end)
    else:
        C = calc_c(n_l, conversion, fr, n_bc, n_fc)

    pm = calc_power_matrix(n)

    X_l = C @ X
    truncated = False

    def objective(rho):
        if method == "chow-lin-maxlog":
            val = -calc_gls(y_l, X_l, C @ calc_q(rho, pm) @ C.T, stats=False)['logl']
        elif method == "chow-lin-minrss-ecotrim":
            val = calc_gls(y_l, X_l, C @ calc_r(rho, pm) @ C.T, logl=False, stats=False)['rss']
        elif method == "chow-lin-minrss-quilis":
            val = calc_gls(y_l, X_l, C @ calc_q(rho, pm) @ C.T, logl=False, stats=False)['rss']
        elif method == "litterman-maxlog":
            val = -calc_gls(y_l, X_l, C @ calc_q_lit(X, rho) @ C.T, stats=False)['logl']
        elif method == "litterman-minrss":
            val = calc_gls(y_l, X_l, C @ calc_q_lit(X, rho) @ C.T, logl=False, stats=False)['rss']
        elif method == "dynamic-maxlog":
            X_adj = calc_dyn_adj(X, rho)
            X_l_adj = C @ X_adj
            val = -calc_gls(y_l, X_l_adj, C @ calc_q(rho, pm) @ C.T, stats=False)['logl']
        elif method == "dynamic-minrss":
            X_adj = calc_dyn_adj(X, rho)
            X_l_adj = C @ X_adj
            val = calc_gls(y_l, X_l_adj, C @ calc_q(rho, pm) @ C.T, logl=False, stats=False)['rss']
        else:
            val = 0
        return val

    if method in [
        "chow-lin-maxlog", "chow-lin-minrss-ecotrim",
        "chow-lin-minrss-quilis", "litterman-maxlog",
        "litterman-minrss", "dynamic-maxlog", "dynamic-minrss"]:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': tol})
        rho = res.x
        if rho < truncated_rho:
            rho = truncated_rho
            truncated = True
    elif method in ["fernandez", "ols"]:
        rho = 0
    elif method in ["chow-lin-fixed", "litterman-fixed", "dynamic-fixed"]:
        rho = fixed_rho
    else:
        rho = 0

    if method in ["fernandez", "litterman-maxlog", "litterman-minrss", "litterman-fixed"]:
        Q = calc_q_lit(X, rho)
    else:
        Q = calc_q(rho, pm)

    if method in ["dynamic-maxlog", "dynamic-minrss", "dynamic-fixed"] and rho != 0:
        X = calc_dyn_adj(X, rho)
        X_l = C @ X

    Q_l = C @ Q @ C.T
    z = calc_gls(y_l, X_l, Q_l)

    if np.linalg.matrix_rank(X) < min(X.shape):
        print("Warning: X is singular!")

    p = X @ z['coefficients']
    D = Q @ C.T @ z['vcov_inv']
    u_l = y_l.reshape(-1, 1) - C @ p.reshape(-1,1)
    y = p.reshape(-1,1) + D @ u_l

    z['vcov_inv'] = None
    z['values'] = y.flatten()
    z['fitted.values'] = (C @ p).flatten()
    z['p'] = p.flatten()
    z['residuals'] = u_l.flatten()
    z['rho'] = rho
    z['truncated'] = truncated
    return z


def sub_denton(y_l, X, lf=None, hf=None, lf_end=None, n_bc=None, n_fc=None,
               conversion="sum", method="denton", fr=None, criterion="proportional", h=1):
    if X.ndim > 1 and X.shape[1] > 1:
        raise ValueError("Right hand side is not a vector, only one series allowed")
    if criterion not in ["additive", "proportional"]:
        raise ValueError("criterion must be additive or proportional")

    if method == "uniform":
        h = 0
        criterion = "additive"
        method = "denton"

    n_l = len(y_l)
    n = X.shape[0]

    if hf is not None:
        C = calc_clfhf(lf, hf, conversion, lf_end)
    else:
        C = calc_c(n_l, conversion, fr, n_bc, n_fc)

    D = np.eye(n)
    D[1:, :-1] -= np.eye(n - 1)
    X_inv = np.diag(1 / (X.flatten() / X.mean()))

    D_0 = D.copy()
    if h == 0:
        if criterion == "proportional":
            D_0 = D_0 @ X_inv
    elif h > 0:
        for _ in range(h):
            D_0 = D @ D_0
        if criterion == "proportional":
            D_0 = D_0 @ X_inv
    else:
        raise ValueError("wrong specification of h")

    u_l = y_l.reshape(-1, 1) - C @ X.reshape(-1, 1)

    if method == "denton-cholette":
        if h == 0:
            D_1 = D_0
        else:
            D_1 = D_0[h:]
        A = D_1.T @ D_1
        mat = np.block([
            [A, C.T],
            [C, np.zeros((n_l, n_l))]
        ])
        y_mat = np.concatenate([X.reshape(-1, 1), u_l], axis=0)
        sol = np.linalg.solve(mat, np.block([
            [A, np.zeros((n, n_l))],
            [C, np.eye(n_l)]
        ]) @ y_mat)
        y = sol[:n]
    elif method == "denton":
        D_1 = D_0
        Q = np.linalg.inv(D_1.T @ D_1)
        Dmat = Q @ C.T @ np.linalg.inv(C @ Q @ C.T)
        y = X.reshape(-1,1) + Dmat @ u_l
    else:
        raise ValueError("wrong method")

    z = {
        'values': y.flatten(),
        'fitted.values': (C @ X.reshape(-1,1)).flatten(),
        'p': X.flatten(),
        'residuals': u_l.flatten(),
        'criterion': criterion,
        'h': h,
    }
    return z

# Additional helper for irregular conversion

def calc_clfhf(lf, hf, conversion, lf_end):
    raise NotImplementedError("Irregular conversion not implemented in Python version")


def td(*args, **kwargs):
    raise NotImplementedError("Full td functionality not implemented. This file provides low-level translations of core algorithms.")
