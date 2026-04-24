import numpy as np
from scipy.optimize import minimize

def optimize_sigma_test(sigma_T, lam, d, squared=False):
    def objective(params):
        L = np.zeros((d, d))
        idx = np.tril_indices(d)
        L[idx] = params
        Sigma = L @ L.T
        det_val = np.linalg.det(Sigma)
        if squared:
            frob = np.sum((Sigma - sigma_T)**2)
        else:
            frob = np.linalg.norm(Sigma - sigma_T, 'fro')
        return det_val + lam * frob

    L0 = np.linalg.cholesky(sigma_T)
    x0 = L0[np.tril_indices(d)]
    
    res = minimize(objective, x0, method='L-BFGS-B')
    L = np.zeros((d, d))
    L[np.tril_indices(d)] = res.x
    return L @ L.T

sigma_T = np.array([[4.0, 0.5], [0.5, 0.3]])

print("Sigma_T:\n", sigma_T)
print("\n--- NON-SQUARED (User's formula) ---")
for lam in [1.0, 5.0, 10.0]:
    s_opt = optimize_sigma_test(sigma_T, lam, 2, squared=False)
    print(f"lam={lam}, Sigma_opt:\n{s_opt}\n")

print("\n--- SQUARED (My formula) ---")
for lam in [1.0, 5.0, 10.0]:
    s_opt = optimize_sigma_test(sigma_T, lam, 2, squared=True)
    print(f"lam={lam}, Sigma_opt:\n{s_opt}\n")
