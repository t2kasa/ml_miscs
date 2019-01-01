import numpy as np


class KalmanFilter:
    def __init__(self, n_states, n_measurements, n_controls):
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.n_controls = n_controls

        self.transition_matrix = np.eye(n_states, n_states)
        self.control_matrix = np.eye(n_states, n_controls)
        self.process_noise_cov = np.eye(n_states, n_states)

        self.observation_matrix = np.eye(n_measurements, n_states)
        self.observation_noise_cov = np.eye(n_measurements, n_measurements)

        self.gain = np.zeros((n_states, n_measurements))
        self.x_pre = np.zeros(n_states)
        self.x_post = np.zeros(n_states)
        self.error_cov_pre = np.zeros((n_states, n_states))
        self.error_cov_post = np.zeros((n_states, n_states))

    def predict(self, control=None):
        self.x_pre = self.transition_matrix @ self.x_post
        if control is not None:
            self.x_pre += self.control_matrix @ control

        self.error_cov_pre = self.process_noise_cov + self.transition_matrix @ self.error_cov_post @ self.transition_matrix.T  # noqa
        return self.x_pre.copy()

    def update(self, measurement):
        # innovation and its covariance
        innovation = measurement - self.observation_matrix @ self.x_pre
        innovation_cov = self.observation_noise_cov + self.observation_matrix @ self.error_cov_pre @ self.observation_matrix.T  # noqa

        self.gain = self.error_cov_pre @ self.observation_matrix.T @ np.linalg.inv(innovation_cov)  # noqa
        self.x_post = self.x_pre + self.gain @ innovation
        self.error_cov_post = (np.eye(self.n_states) - self.gain @ self.observation_matrix) @ self.error_cov_pre  # noqa
        return self.x_post.copy()

    def estimate_measurement(self, x):
        return self.observation_matrix @ x
