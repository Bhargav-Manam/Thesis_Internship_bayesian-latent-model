import numpy as np
import matplotlib.pyplot as plt
from utils import helper_functions as hf


class Generator:
    def __init__(self, n, d, q, min_len, max_len, max_time):
        self.metadata = {
            'n': n,
            'd': d,
            'q': q,
            'min_len': min_len,
            'max_len': max_len,
            'max_time': max_time,
        }
        self.attributes = {}
        self.params = {}
        self.generate_attributes()
        self.generate_params()

    def generate_attributes(self, seed=673):
        self.attributes['t'] = hf.generate_t(self.metadata['n'], self.metadata['d'], self.metadata['min_len'],
                                             self.metadata['max_len'], self.metadata['max_time'], seed=seed)

    def generate_params(self, seed=908):
        self.params['mu'], self.params['omega'], self.params['w'], self.params['v'], self.params[
            'sigma2'] = hf.random_params(self.metadata['d'], self.metadata['q'], seed=seed)

    def plot_t(self, subject_indices=None):
        d = self.metadata['d']
        n = self.metadata['n']
        max_time = self.metadata['max_time']
        t = self.attributes['t']

        cmap = plt.get_cmap('tab10')
        num_modalities = len(d)
        colors = cmap(np.linspace(0, 1, num_modalities))

        if subject_indices is None:
            subject_indices = range(n)

        for subject in subject_indices:
            fig, ax = plt.subplots(figsize=(10, 3))

            for modality in range(num_modalities):
                time_sequence = t[subject][modality]
                y = [modality + 1] * len(time_sequence)
                ax.scatter(time_sequence, y, color=colors[modality], alpha=1)

            ax.set_xlabel('Time')
            ax.set_ylabel('Modalities')
            ax.set_title('Subject {}'.format(subject + 1))
            ax.set_xlim(-0.5, max_time - 0.5)
            ax.set_ylim(0.5, num_modalities + 0.5)
            ax.set_xticks(range(max_time))
            ax.set_yticks(range(1, num_modalities + 1))
            ax.set_yticklabels(['Modality {}'.format(i) for i in range(1, num_modalities + 1)])
            ax.grid(True)

        plt.show()

    def generate_data(self, gaussian=True, gaussian_prob=0.5, noise=True, seed=1228):
        d = self.metadata['d']
        n = self.metadata['n']
        t = self.attributes['t']
        w = self.params['w']
        v = self.params['v']
        omega = self.params['omega']
        mu = self.params['mu']
        sigma2 = self.params['sigma2']

        x = hf.sample_x(self.metadata['q'], self.metadata['n'], gaussian=gaussian, gaussian_prob=gaussian_prob,
                        seed=seed)
        y = []
        for i in range(n):
            y_n = []
            for k in range(len(d)):
                y_nk = ((t[i][k].reshape(-1, 1, 1) * w[k] + v[k]) * x[i]).sum(2) + t[i][k].reshape(-1, 1) * omega[k] + \
                       mu[k]
                if noise:
                    epsilon = np.array([])
                    for j in range(len(t[i][k])):
                        if seed is not None:
                            rnd = np.random.RandomState(seed + i + k + j)
                        else:
                            rnd = np.random.default_rng()
                        eps = rnd.multivariate_normal(np.zeros(d[k]),
                                                      np.eye(d[k]) * sigma2[k])
                        epsilon = np.append(epsilon, eps)
                    epsilon = epsilon.reshape(len(t[i][k]), d[k])
                    y_nk += epsilon
                y_n.append(y_nk)
            y.append(y_n)
        return x, y
