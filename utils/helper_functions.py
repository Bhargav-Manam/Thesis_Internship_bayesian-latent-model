import numpy as np
import matplotlib.pyplot as plt
import itertools


def fit(y, w, v, mu, omega, sigma2, t, n_epochs=200, history=True):
    T, eta, tau = compute_time_vars(t)

    if history:
        training_history = np.zeros((1, n_epochs))
    else:
        training_history = None

    for i in range(n_epochs):
        i_mom, ii_mom, _ = posterior_moments(y, w, v, mu, omega, sigma2, t)
        mu = mu_opt(y, w, v, omega, i_mom, T, eta)
        omega = omega_opt(y, w, v, mu, i_mom, eta, tau, t)
        w = w_opt(y, v, mu, omega, i_mom, ii_mom, eta, tau, t)
        v = v_opt(y, w, mu, omega, i_mom, ii_mom, T, eta)
        sigma2 = sigma2_opt(y, w, v, mu, omega, i_mom, ii_mom, T, t)

        if history:
            training_history[0, i] = marginal_likelihood(y, w, v, mu, omega, sigma2, t)

    return w, v, mu, omega, sigma2, training_history


def generate_t(n, d, min_len, max_len, max_time, seed=383):
    t = []
    for j in range(n):
        subject_seq = []
        for i in range(len(d)):
            if seed is not None:
                rnd = np.random.RandomState(28 + i + j)
            else:
                rnd = np.random.default_rng()
            sequence_len = rnd.randint(min_len, max_len + 1)
            sequence = np.sort(rnd.choice(range(max_time), size=sequence_len, replace=False))
            subject_seq.append(sequence)
        t.append(subject_seq)

    return t


def random_params(d, q, seed=47):
    mu, omega, w, v, sigma2 = [], [], [], [], []

    for k in range(len(d)):
        if seed is not None:
            rnd = np.random.RandomState(seed + k)
        else:
            rnd = np.random.default_rng()
        mu_k = rnd.uniform(-1, 1, d[k])
        omega_k = rnd.uniform(-1, 1, d[k])
        w_k = rnd.uniform(-1, 1, (d[k], q))
        v_k = rnd.uniform(-1, 1, (d[k], q))
        sigma2_k = rnd.uniform(1, 2)
        mu.append(mu_k)
        omega.append(omega_k)
        w.append(w_k)
        v.append(v_k)
        sigma2.append(sigma2_k)

    return mu, omega, w, v, sigma2


def sample_x(q, n, gaussian=True, gaussian_prob=0.5, seed=52):
    if seed is not None:
        rnd = np.random.RandomState(seed)
    else:
        rnd = np.random.default_rng()

    if gaussian:
        return rnd.multivariate_normal(mean=np.zeros(q), cov=np.eye(q), size=n)
    else:
        samples = np.zeros((n, q))
        for i in range(n):
            if rnd.random() < gaussian_prob:
                samples[i] = rnd.multivariate_normal(mean=np.zeros(q), cov=np.eye(q))
            else:
                samples[i] = rnd.uniform(low=-2, high=2, size=q)

        return samples


def decode(x, w, v, mu, omega, sigma2, t, seed=78, noise=True):
    n = len(x)
    y = []
    for i in range(n):
        y_n = []
        for k in range(len(mu)):
            y_nk = ((t[i][k].reshape(-1, 1, 1) * w[k] + v[k]) * x[i]).sum(2) + t[i][k].reshape(-1, 1) * omega[k] + mu[k]
            if noise:
                epsilon = np.array([])
                for j in range(len(t[i][k])):
                    if seed is not None:
                        rnd = np.random.RandomState(seed + i + j + k)
                    else:
                        rnd = np.random.default_rng()
                    eps = rnd.multivariate_normal(np.zeros(len(mu[k])),
                                                  np.eye(len(mu[k])) * sigma2[k])
                    epsilon = np.append(epsilon, eps)
                epsilon = epsilon.reshape(len(t[i][k]), len(mu[k]))
                y_nk += epsilon
            y_n.append(y_nk)
        y.append(y_n)

    return y


def encode(y, w, v, mu, omega, sigma2, t, seed=128):
    n = len(y)
    _, q = w[0].shape
    x_mean, _, x_variance = posterior_moments(y, w, v, mu, omega, sigma2, t)
    x = np.zeros((n, q))
    for i in range(n):
        if seed is not None:
            rnd = np.random.RandomState(seed + i)
        else:
            rnd = np.random.default_rng()
        x[i] = rnd.multivariate_normal(mean=x_mean[i], cov=x_variance[i])
    return x, x_mean, x_variance


def compute_time_vars(t):
    n = len(t)
    modalities = len(t[0])
    T, eta, tau = [], [], []

    for i in range(n):
        T_k, eta_k, tau_k = [], [], []
        for k in range(modalities):
            t_size = t[i][k].size
            t_sum = np.sum(t[i][k])
            t_square_sum = np.sum(t[i][k] * t[i][k])
            T_k.append(t_size), eta_k.append(t_sum), tau_k.append(t_square_sum)
        T.append(T_k), eta.append(eta_k), tau.append(tau_k)

    return T, eta, tau


def posterior_moments(y, w, v, mu, omega, sigma2, t):
    n = len(y)
    _, q = w[0].shape
    i_mom = np.zeros((n, q), dtype=np.float64)
    ii_mom = np.zeros((n, q, q), dtype=np.float64)
    variance = np.zeros((n, q, q), dtype=np.float64)
    T, eta, tau = compute_time_vars(t)

    w2, v2, cross = [], [], []
    for i in range(len(mu)):
        w2.append(w[i].T @ w[i])
        v2.append(v[i].T @ v[i])
        temp_cross = w[i].T @ v[i]
        cross.append(temp_cross + temp_cross.T)

    for j in range(n):
        temp_variance = np.zeros((1, q, q), dtype=np.float64)
        temp_i_mom = np.zeros((1, q, 1), dtype=np.float64)
        for k in range(len(mu)):
            temp_variance += (tau[j][k] * w2[k] + eta[j][k] * cross[k] + T[j][k] * v2[k]) * (sigma2[k] ** -1)
            temp_param = t[j][k].reshape(T[j][k], 1, 1) * w[k].reshape(1, len(mu[k]), q) + v[k].reshape(1, len(mu[k]),
                                                                                                        q)
            temp_1 = (y[j][k] - t[j][k].reshape(-1, 1) * omega[k] - mu[k]).reshape(T[j][k], len(mu[k]), 1)
            temp_i_mom += (sigma2[k] ** -1) * ((np.transpose(temp_param, axes=(0, 2, 1)) @ temp_1).sum(0))
        variance[j] = np.linalg.inv(temp_variance + np.eye(q))
        i_mom[j] = (variance[j] @ temp_i_mom).reshape(q, )
        ii_mom[j] = variance[j] + i_mom[j].reshape(q, 1) * i_mom[j].reshape(1, q)

    return i_mom, ii_mom, variance


def marginal_likelihood(y, w, v, mu, omega, sigma2, t):
    n = len(y)
    _, q = w[0].shape
    T, eta, tau = compute_time_vars(t)
    i_mom, ii_mom, _ = posterior_moments(y, w, v, mu, omega, sigma2, t)
    likelihood = 0.0

    for j in range(n):
        for k in range(len(mu)):
            term_1 = T[j][k] * len(mu[k]) / 2 * np.log(sigma2[k])

            temp_param_1 = (y[j][k] - t[j][k].reshape(-1, 1) * omega[k] - mu[k]).reshape(T[j][k] * len(mu[k]), 1)
            temp_param_2 = t[j][k].reshape(T[j][k], 1, 1) * w[k].reshape(1, len(mu[k]), q) + v[k].reshape(1, len(mu[k]),
                                                                                                          q)
            temp_param_2 = temp_param_2.reshape(T[j][k] * len(mu[k]), q)
            temp_param_3 = -2 * i_mom[j].T @ temp_param_2.T @ temp_param_1
            temp_param_1 = temp_param_1.T @ temp_param_1
            temp_param_2 = np.trace(temp_param_2.T @ temp_param_2 @ ii_mom[j])

            term_2 = (1 / (2 * sigma2[k])) * (temp_param_1 + temp_param_2 + temp_param_3)
            term_3 = 0.5 * np.trace(ii_mom[j])
            likelihood += -1 * (term_1 + term_2 + term_3)

    return likelihood


def mu_opt(y, w, v, omega, i_mom, T, eta):
    n = len(y)
    T = np.array(T)
    T_sum = np.sum(T, axis=0)
    mu = []
    for k in range(len(omega)):
        mu.append([])
        mu[k] = np.zeros(len(omega[k]), dtype=np.float64)
        for j in range(n):
            mu[k] += np.sum(y[j][k], axis=0) - (eta[j][k] * w[k] + T[j][k] * v[k]) @ i_mom[j] - eta[j][k] * omega[k]
        mu[k] = np.array(mu[k]) / T_sum[k]

    return mu


def omega_opt(y, w, v, mu, i_mom, eta, tau, t):
    n = len(y)
    tau = np.array(tau)
    tau_sum = np.sum(tau, axis=0)
    omega = []
    for k in range(len(mu)):
        omega.append([])
        omega[k] = np.zeros(len(mu[k]), dtype=np.float64)
        for j in range(n):
            omega[k] += np.sum(t[j][k].reshape(-1, 1) * y[j][k], axis=0) - (tau[j][k] * w[k] + eta[j][k] * v[k]) @ \
                        i_mom[j] - eta[j][k] * mu[k]
        omega[k] = np.array(omega[k]) / tau_sum[k]

    return omega


def w_opt(y, v, mu, omega, i_mom, ii_mom, eta, tau, t):
    n = len(y)
    _, q = v[0].shape
    w = []

    for k in range(len(mu)):
        at = np.zeros((q, len(mu[k])), dtype=np.float64)
        bt = np.zeros((q, q), dtype=np.float64)
        for j in range(n):
            at += i_mom[j].reshape(q, 1) * ((t[j][k].reshape(-1, 1) * y[j][k]).sum(0) - tau[j][k] * omega[k]
                                            - eta[j][k] * mu[k]).reshape(1, len(mu[k])) - eta[j][k] * (
                          ii_mom[j].T @ v[k].T)
            bt += tau[j][k] * ii_mom[j].T
        w.append([])
        w[k] = np.linalg.solve(bt, at).T

    return w


def v_opt(y, w, mu, omega, i_mom, ii_mom, T, eta):
    n = len(y)
    _, q = w[0].shape
    v = []

    for k in range(len(mu)):
        at = np.zeros((q, len(mu[k])), dtype=np.float64)
        bt = np.zeros((q, q), dtype=np.float64)
        for j in range(n):
            at += i_mom[j].reshape(q, 1) * (y[j][k].sum(0) - eta[j][k] * omega[k] - T[j][k] * mu[k]).reshape(1, len(
                mu[k])) - eta[j][k] * (ii_mom[j].T @ w[k].T)
            bt += T[j][k] * ii_mom[j].T
        v.append([])
        v[k] = np.linalg.solve(bt, at).T

    return v


def sigma2_opt(y, w, v, mu, omega, i_mom, ii_mom, T, t):
    n = len(y)
    _, q = w[0].shape
    T_sum = np.sum(T, axis=0)
    sigma2 = []
    for k in range(len(mu)):
        sigma2.append([])
        sigma2[k] = 0.0
        for j in range(n):
            temp_param = t[j][k].reshape(T[j][k], 1, 1) * w[k].reshape(1, len(mu[k]), q) + v[k].reshape(1, len(mu[k]),
                                                                                                        q)
            temp = y[j][k] - t[j][k].reshape(-1, 1) * omega[k] - mu[k]

            sigma2[k] += ((temp * temp).sum() + (
                (((temp_param.reshape(T[j][k], len(mu[k]), q, 1) * temp_param.reshape(T[j][k], len(mu[k]), 1, q)).sum(
                    axis=1)
                  .reshape(T[j][k], q, q, 1) * ii_mom[j].reshape(1, 1, q, q)).sum(2)).reshape(T[j][k], q * q)[:,
                0:q * q:q + 1]).sum() - 2 * (
                                  (i_mom[j].reshape(1, 1, q) * temp_param).sum(axis=2).reshape(T[j][k], len(mu[k]), 1) *
                                  temp.reshape(T[j][k], len(mu[k]), 1)).sum())
        sigma2[k] = sigma2[k] / (len(mu[k]) * T_sum[k])

    return sigma2


def mae(y_true, y_pred):
    n = len(y_true)
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    max_modalities = max(len(y_true[j]) for j in range(len(y_true)))
    mae_modalities = np.zeros(max_modalities)

    for j in range(n):
        modalities = len(y_true[j])
        for k in range(modalities):
            mae_temp = np.mean(np.abs(y_true[j][k] - y_pred[j][k]))
            mae_modalities[k] += mae_temp
    mae_modalities /= n
    return np.sum(mae_modalities), mae_modalities


def plot_scatter_pairs_latent(data, point_size=10):
    num_points, num_dimensions = data.shape
    dimension_pairs = list(itertools.combinations(range(num_dimensions), 2))

    num_plots = len(dimension_pairs)
    plt.figure(figsize=(5 * num_plots, 5))

    for idx, (dim1, dim2) in enumerate(dimension_pairs):
        plt.subplot(1, num_plots, idx + 1)
        plt.scatter(data[:, dim1], data[:, dim2], s=point_size)
        plt.xlabel(f'Dimension {dim1 + 1}')
        plt.ylabel(f'Dimension {dim2 + 1}')
        plt.title(f'Scatter plot of dimensions {dim1 + 1} and {dim2 + 1}')

    plt.tight_layout()
    plt.show()


def plot_scatter_pairs_observed(data, subject_indices=None, point_size=10):
    data = np.array(data, dtype="object")
    _, modality_index = data.shape

    for modality_index in range(modality_index):
        if subject_indices is None:
            subjects_data = np.concatenate([subject[modality_index] for subject in data])
            subject_info = 'all subjects'
        else:
            subjects_data = np.concatenate([data[i][modality_index] for i in subject_indices])
            subject_info = f'subjects {subject_indices}'

        num_points, num_dimensions = subjects_data.shape
        dimension_pairs = list(itertools.combinations(range(num_dimensions), 2))

        num_plots = len(dimension_pairs)
        plt.figure(figsize=(5 * num_plots, 5))

        for idx, (dim1, dim2) in enumerate(dimension_pairs):
            plt.subplot(1, num_plots, idx + 1)
            plt.scatter(subjects_data[:, dim1], subjects_data[:, dim2], s=point_size)
            plt.xlabel(f'Dimension {dim1 + 1}')
            plt.ylabel(f'Dimension {dim2 + 1}')
            plt.title(
                f'Plot of dimensions {dim1 + 1} and {dim2 + 1} for modality {modality_index + 1}, {subject_info}')

        plt.tight_layout()
        plt.show()


def plot_likelihood(likelihood):
    _, n_epochs = likelihood.shape
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, likelihood[0], marker='o')
    plt.title('Likelihood Convergence Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Likelihood')
    plt.grid(True)
    plt.show()
