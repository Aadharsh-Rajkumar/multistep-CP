import numpy as np

def generate_drone_trajectory(T=2000, shift_t=1000, dt=0.1):
    """
    Simulates a 2D drone tracking a reference Lissajous curve.

    Returns:
        positions: (T,2) noisy observed positions
        references: (T,2) ground-truth reference trajectory
    """

    np.random.seed(42)

    p = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])

    A_x, A_y = 5.0, 3.0
    omega_x, omega_y = 0.05, 0.07

    Kp = 2.0
    Kd = 1.5

    positions = np.zeros((T, 2))
    references = np.zeros((T, 2))

    for t in range(T):

        t_sec = t * dt

        p_ref = np.array([
            A_x * np.sin(omega_x * t_sec),
            A_y * np.sin(omega_y * t_sec)
        ])

        v_ref = np.array([
            A_x * omega_x * np.cos(omega_x * t_sec),
            A_y * omega_y * np.cos(omega_y * t_sec)
        ])

        references[t] = p_ref

        u = -Kp * (p - p_ref) - Kd * (v - v_ref)

        if t >= shift_t:
            degradation_matrix = np.array([
                [0.4, 0.1],
                [0.1, 0.2]
            ])
            u = degradation_matrix @ u

        process_noise = np.random.multivariate_normal(
            [0, 0],
            [[0.01, 0.0],
             [0.0, 0.01]]
        )

        v = v + u * dt + process_noise
        p = p + v * dt

        obs_noise = np.random.multivariate_normal(
            [0, 0],
            [[0.005, 0.0],
             [0.0, 0.005]]
        )

        positions[t] = p + obs_noise

    return positions, references