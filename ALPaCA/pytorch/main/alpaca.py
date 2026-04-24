import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy


class ALPaCA(nn.Module):
    def __init__(self, config, device=None, preprocess=None, f_nom=None):
        super(ALPaCA, self).__init__()
        self.config = deepcopy(config)
        self.lr = config['lr']
        self.x_dim = config['x_dim']
        self.phi_dim = config['nn_layers'][-1]
        self.y_dim = config['y_dim']
        self.sigma_eps = self.config['sigma_eps']

        self.updates_so_far = 0
        self.device = device if device is not None else torch.device('cpu')

        # y = K^T phi( preprocess(x) ) + f_nom(x)
        self.preprocess = preprocess
        self.f_nom = f_nom

    def construct_model(self):
        last_layer = self.config['nn_layers'][-1]

        if isinstance(self.sigma_eps, list):
            self.register_buffer('SigEps', torch.diag(torch.tensor(self.sigma_eps, dtype=torch.float32)))
        else:
            self.register_buffer('SigEps', self.sigma_eps * torch.eye(self.y_dim))
        self.SigEps = self.SigEps.reshape(1, 1, self.y_dim, self.y_dim)

        # Prior Parameters of last layer
        self.K = nn.Parameter(torch.randn(last_layer, self.y_dim) * 0.01)  # \bar{K}_0
        self.L_asym = nn.Parameter(torch.randn(last_layer, last_layer) * 0.01)  # cholesky decomp of \Lambda_0

        # Build basis network
        self._build_basis_network()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def _build_basis_network(self):
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        act_cls = activations[self.config['activation']]

        layers = []
        inp_dim = self.x_dim
        for units in layer_sizes:
            layers.append(nn.Linear(inp_dim, units))
            layers.append(act_cls())
            inp_dim = units
        self.basis_net = nn.Sequential(*layers)

    @property
    def L(self):
        return self.L_asym @ self.L_asym.T

    def basis(self, x):
        """Map input x (..., x_dim) through the basis network."""
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self.basis_net(x)

    def batch_blr(self, X, Y, num):
        """Bayesian linear regression update for a single task."""
        X = X[:num, :]
        Y = Y[:num, :]
        L = self.L
        if num > 0:
            Ln_inv = torch.linalg.inv(X.T @ X + L)
            Kn = Ln_inv @ (X.T @ Y + L @ self.K)
        else:
            Kn = self.K.clone()
            Ln_inv = torch.linalg.inv(L)
        return Kn, Ln_inv

    def compute_posterior(self, context_phi, context_y_blr, num_context):
        """Compute posterior K and L_inv for each task in the batch."""
        M = context_phi.shape[0]
        posterior_K = torch.zeros(M, self.phi_dim, self.y_dim, device=self.device)
        posterior_L_inv = torch.zeros(M, self.phi_dim, self.phi_dim, device=self.device)

        for i in range(M):
            posterior_K[i], posterior_L_inv[i] = self.batch_blr(
                context_phi[i], context_y_blr[i], num_context[i]
            )
        return posterior_K, posterior_L_inv

    def compute_pred_and_nll(self, phi, posterior_K, posterior_L_inv, y, f_nom_x):
        """
        Compute posterior predictive distribution and NLL.
        Args:
            phi: (M, T, phi_dim)
            posterior_K: (M, phi_dim, y_dim)
            posterior_L_inv: (M, phi_dim, phi_dim)
            y: (M, T, y_dim)
            f_nom_x: scalar 0 or (M, T, y_dim)
        Returns:
            mu_pred: (M, T, y_dim)
            Sig_pred: (M, T, y_dim, y_dim)
            predictive_nll: (M, T)
        """
        # mu_pred = K^T @ phi + f_nom
        mu_pred = batch_matmul(posterior_K.transpose(-1, -2), phi) + f_nom_x

        # spread_fac = 1 + phi^T L_inv phi
        spread_fac = 1 + batch_quadform(posterior_L_inv, phi)

        SigEps = self.SigEps.to(self.device)
        Sig_pred = spread_fac.unsqueeze(-1) * SigEps.reshape(1, 1, self.y_dim, self.y_dim)

        # NLL
        logdet = self.y_dim * torch.log(spread_fac) + torch.linalg.slogdet(SigEps)[1]
        Sig_pred_inv = torch.linalg.inv(Sig_pred)
        quadf = batch_quadform(Sig_pred_inv, (y - mu_pred))

        predictive_nll = (logdet + quadf).squeeze(-1)

        return mu_pred, Sig_pred, predictive_nll

    def forward(self, context_x, context_y, x, y, num_context=None):
        """
        Full forward pass: compute basis, BLR, predictions, and loss.
        """
        M = context_x.shape[0]

        if num_context is None:
            num_context = torch.full((M,), context_x.shape[1], dtype=torch.long)

        # Map to feature space
        phi = self.basis(x)
        context_phi = self.basis(context_x)

        # Evaluate f_nom
        f_nom_cx = torch.zeros_like(context_y)
        f_nom_x = 0
        if self.f_nom is not None:
            f_nom_cx = self.f_nom(context_x)
            f_nom_x = self.f_nom(x)

        context_y_blr = context_y - f_nom_cx

        # Posterior
        posterior_K, posterior_L_inv = self.compute_posterior(context_phi, context_y_blr, num_context)

        # Predictions and NLL
        mu_pred, Sig_pred, predictive_nll = self.compute_pred_and_nll(
            phi, posterior_K, posterior_L_inv, y, f_nom_x
        )

        total_loss = predictive_nll.mean()
        return mu_pred, Sig_pred, total_loss

    # ---- Train and Test functions ------ #
    def train_model(self, dataset, num_train_updates):
        self.train()
        batch_size = self.config['meta_batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']

        for i in range(num_train_updates):
            x, y = dataset.sample(n_funcs=batch_size, n_samples=horizon + test_horizon)

            context_x = torch.tensor(x[:, :horizon, :], dtype=torch.float32, device=self.device)
            context_y = torch.tensor(y[:, :horizon, :], dtype=torch.float32, device=self.device)
            test_x = torch.tensor(x[:, horizon:, :], dtype=torch.float32, device=self.device)
            test_y = torch.tensor(y[:, horizon:, :], dtype=torch.float32, device=self.device)
            num_context = torch.randint(0, horizon + 1, (batch_size,))

            self.optimizer.zero_grad()
            _, _, loss = self.forward(context_x, context_y, test_x, test_y, num_context)
            loss.backward()
            self.optimizer.step()

            if i % 50 == 0:
                print('loss:', loss.item())

            self.updates_so_far += 1

    # x_c, y_c, x are all numpy arrays [N, n]
    # returns mu_pred, Sig_pred as numpy arrays
    def test(self, x_c, y_c, x):
        self.eval()
        with torch.no_grad():
            x_c_t = torch.tensor(x_c, dtype=torch.float32, device=self.device)
            y_c_t = torch.tensor(y_c, dtype=torch.float32, device=self.device)
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

            M = x_c_t.shape[0]

            phi = self.basis(x_t)
            context_phi = self.basis(x_c_t)

            f_nom_cx = torch.zeros_like(y_c_t)
            f_nom_x = 0
            if self.f_nom is not None:
                f_nom_cx = self.f_nom(x_c_t)
                f_nom_x = self.f_nom(x_t)

            context_y_blr = y_c_t - f_nom_cx
            num_context = torch.full((M,), x_c_t.shape[1], dtype=torch.long)

            posterior_K, posterior_L_inv = self.compute_posterior(context_phi, context_y_blr, num_context)

            mu_pred = batch_matmul(posterior_K.transpose(-1, -2), phi) + f_nom_x
            spread_fac = 1 + batch_quadform(posterior_L_inv, phi)
            SigEps = self.SigEps.to(self.device)
            Sig_pred = spread_fac.unsqueeze(-1) * SigEps.reshape(1, 1, self.y_dim, self.y_dim)

        return mu_pred.cpu().numpy(), Sig_pred.cpu().numpy()

    # convenience function to use just the encoder on numpy input
    def encode(self, x):
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            phi = self.basis(x_t)
        return phi.cpu().numpy()

    # ---- Save and Restore ------
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        print('Saved to:', model_path)

    def restore(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        print('Restored model from:', model_path)


# given mat [a,b,c,...,N,N] and batch_v [a,b,c,...,M,N], returns [a,b,c,...,M,N]
def batch_matmul(mat, batch_v):
    return (mat @ batch_v.transpose(-1, -2)).transpose(-1, -2)


# works for A = [...,n,n] or [...,N,n,n]
# assumes b = [...,N,n]
# returns  [...,N,1]
def batch_quadform(A, b):
    A_dims = A.dim()
    b_dims = b.dim()
    b_vec = b.unsqueeze(-1)
    if A_dims == b_dims + 1:
        return (b_vec.transpose(-1, -2) @ A @ b_vec).squeeze(-1)
    elif A_dims == b_dims:
        Ab = (A @ b.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return (b_vec.transpose(-1, -2) @ Ab).squeeze(-1)
    else:
        raise ValueError('Matrix size of %d is not supported.' % (A_dims))


# takes in y = (..., y_dim), x = (..., x_dim)
# returns dydx = (..., y_dim, x_dim), the jacobian of y wrt x
def batch_2d_jacobian(y, x):
    """Compute batch Jacobian dy/dx using torch autograd."""
    y_dim = y.shape[-1]
    x_dim = x.shape[-1]
    batch_shape = y.shape[:-1]

    jac = torch.zeros(*batch_shape, y_dim, x_dim, device=y.device)
    for i in range(y_dim):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[..., i] = 1.0
        grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        jac[..., i, :] = grads
    return jac


#------------ END General ALPaCA -------------#

def blr_update_np(K, L, X, Y):
    Ln_inv = np.linalg.inv(X.T @ X + L)
    Kn = Ln_inv @ (X.T @ Y + L @ K)
    return Kn, Ln_inv


def sampleMN(K, L_inv, Sig):
    mean = np.reshape(K.T, [-1])
    cov = np.kron(Sig, L_inv)
    K_vec = np.random.multivariate_normal(mean, cov)
    return np.reshape(K_vec, K.T.shape).T


def tp(x):
    return np.swapaxes(x, -1, -2)


class AdaptiveDynamics(ALPaCA):
    def __init__(self, config, device=None, uncertainty_prop=False, preprocess=None, f_nom=None):
        if f_nom is None:
            y_dim = config['y_dim']
            f_nom = lambda xu: xu[..., :y_dim]

        super(AdaptiveDynamics, self).__init__(config, device, preprocess, f_nom)
        self.uncertainty_prop = uncertainty_prop

    def construct_model(self):
        super(AdaptiveDynamics, self).construct_model()

    def train_model(self, dataset, num_train_updates):
        super(AdaptiveDynamics, self).train_model(dataset, num_train_updates)

        # obtain parameters of belief over K as np arrays
        with torch.no_grad():
            self.K0 = self.K.cpu().numpy().copy()
            self.L0 = self.L.cpu().numpy().copy()
            self.sigeps = self.SigEps[0, 0, :, :].cpu().numpy().copy()
        self.reset_to_prior()

    def restore(self, model_path):
        super(AdaptiveDynamics, self).restore(model_path)

        with torch.no_grad():
            self.K0 = self.K.cpu().numpy().copy()
            self.L0 = self.L.cpu().numpy().copy()
            self.sigeps = self.SigEps[0, 0, :, :].cpu().numpy().copy()
        self.reset_to_prior()

    # ---- Utilities for downstream use of the dynamics model -------
    def encode(self, x, return_grad=False):
        if not return_grad:
            self.eval()
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
                phi = self.basis(x_t)
            return phi.cpu().numpy()
        else:
            self.eval()
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)
            phi = self.basis(x_t)

            # Compute Jacobian dphi/dx
            phi_dim = phi.shape[-1]
            x_dim = x_t.shape[-1]
            batch_shape = phi.shape[:-1]

            dphi_dxu = torch.zeros(*batch_shape, phi_dim, x_dim, device=self.device)
            for i in range(phi_dim):
                grad_outputs = torch.zeros_like(phi)
                grad_outputs[..., i] = 1.0
                grads = torch.autograd.grad(phi, x_t, grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]
                dphi_dxu[..., i, :] = grads

            return phi.detach().cpu().numpy(), dphi_dxu.detach().cpu().numpy()

    def gen_predict_func(self):
        K = self.Ln_inv @ self.Qn

        def predict(x, u):
            x_inp = np.reshape(np.concatenate((x, u)), (1, 1, -1))
            phi = self.encode(x_inp, return_grad=False)
            phi = phi[0, 0, :]
            f_nom_val = x_inp[0, 0, :self.config['y_dim']]  # f_nom = identity on x
            return f_nom_val + K.T @ phi

        return predict

    def gen_linearize_func(self):
        K = self.Ln_inv @ self.Qn

        def linearize(x, u):
            batch_size = x.shape[0]
            x_dim = x.shape[-1]
            u_dim = u.shape[-1]
            x_inp = np.reshape(np.concatenate((x, u), axis=-1), (-1, 1, x_dim + u_dim))
            phi, dphi_dxu = self.encode(x_inp, return_grad=True)
            phi = phi[:, 0, :]
            Ap = dphi_dxu[:, 0, :, :x_dim]
            Bp = dphi_dxu[:, 0, :, x_dim:]

            S = np.zeros((batch_size, 1 + x_dim + u_dim, 1 + x_dim + u_dim))
            Sig = np.zeros((batch_size, x_dim, x_dim))
            A = np.zeros((batch_size, x_dim, x_dim))
            B = np.zeros((batch_size, x_dim, u_dim))

            for i in range(batch_size):
                s = (1 + phi[i, :].T @ self.Ln_inv @ phi[i, :])
                Sx = Ap[i, :, :].T @ self.Ln_inv @ phi[i, :]
                Su = Bp[i, :, :].T @ self.Ln_inv @ phi[i, :]
                Sxx = Ap[i, :, :].T @ self.Ln_inv @ Ap[i, :, :]
                Suu = Bp[i, :, :].T @ self.Ln_inv @ Bp[i, :, :]
                Sux = Bp[i, :, :].T @ self.Ln_inv @ Ap[i, :, :]

                S[i, 0, 0] = s
                S[i, 1:1 + x_dim, 0] = Sx
                S[i, 0, 1:1 + x_dim] = Sx.T
                S[i, 1 + x_dim:, 0] = Su
                S[i, 0, 1 + x_dim:] = Su.T
                S[i, 1:1 + x_dim, 1:1 + x_dim] = Sxx
                S[i, 1:1 + x_dim, 1 + x_dim:] = Sux.T
                S[i, 1 + x_dim:, 1:1 + x_dim] = Sux
                S[i, 1 + x_dim:, 1 + x_dim:] = Suu

                A[i, :, :] = K.T @ Ap[i, :, :]
                B[i, :, :] = K.T @ Bp[i, :, :]

            Sig[:, :, :] = np.reshape(self.sigeps, (1, x_dim, x_dim))

            return Sig, S, A + np.eye(x_dim), B

        return linearize

    def gen_dynamics(self, sample_model=False):
        K = self.Ln_inv @ self.Qn
        if sample_model:
            K = sampleMN(K, self.Ln_inv, self.sigeps)

        def dynamics(x, u):
            x_dim = x.shape[-1]
            u_dim = u.shape[-1]
            x_inp = np.reshape(np.concatenate((x, u)), (1, 1, -1))
            phi, dphi_dxu = self.encode(x_inp, return_grad=True)
            phi = phi[0, 0, :]
            A = dphi_dxu[0, 0, :, :x_dim]
            B = dphi_dxu[0, 0, :, x_dim:]
            S = np.zeros((1 + x_dim + u_dim, 1 + x_dim + u_dim))
            if sample_model:
                S[0, 0] = 1.
            else:
                s = (1 + phi.T @ self.Ln_inv @ phi)
                Sx = A.T @ self.Ln_inv @ phi
                Su = B.T @ self.Ln_inv @ phi
                Sxx = A.T @ self.Ln_inv @ A
                Suu = B.T @ self.Ln_inv @ B
                Sux = B.T @ self.Ln_inv @ A

                S[0, 0] = s
                S[1:1 + x_dim, 0] = Sx
                S[0, 1:1 + x_dim] = Sx.T
                S[1 + x_dim:, 0] = Su
                S[0, 1 + x_dim:] = Su.T
                S[1:1 + x_dim, 1:1 + x_dim] = Sxx
                S[1:1 + x_dim, 1 + x_dim:] = Sux.T
                S[1 + x_dim:, 1:1 + x_dim] = Sux
                S[1 + x_dim:, 1 + x_dim:] = Suu

            Sig = self.sigeps
            xp = x + K.T @ phi

            return xp, Sig, S, K.T @ A + np.eye(x_dim), K.T @ B

        return dynamics

    # returns the error of a transition prediction under each model K
    def error(self, x, u, xp, Ks):
        x_inp = np.reshape(np.concatenate((x, u)), (1, 1, -1))
        phi = self.encode(x_inp, return_grad=False)
        phi = phi.reshape((1, -1, 1))
        xp_preds = np.squeeze(tp(Ks) @ phi, axis=-1)
        mses = np.sum((xp_preds - xp.reshape((1, -1))) ** 2, axis=-1)
        return mses

    # returns a list of N = num_models model parameter samples K1 .. KN
    def sample_dynamics_matrices(self, n_models=10):
        return np.array([sampleMN(self.Ln_inv @ self.Qn, self.Ln_inv, self.sigeps) for _ in range(n_models)])

    def sample_rollout(self, x0, actions, Ks, resample=False):
        T, u_dim = actions.shape
        x_dim = x0.shape[0]
        N_samples = len(Ks)

        actions = np.tile(np.expand_dims(actions, axis=0), (N_samples, 1, 1))
        x0 = np.expand_dims(x0, axis=0)

        x_pred = np.zeros((N_samples, T + 1, x_dim))
        x_pred[:, 0, :] = x0

        A = np.zeros((N_samples, T, x_dim, x_dim))
        B = np.zeros((N_samples, T, x_dim, u_dim))

        for t in range(0, T):
            x_inp = np.concatenate((x_pred[:, t:t + 1, :], actions[:, t:t + 1, :]), axis=2)
            phi, dphi_dxu = self.encode(x_inp, return_grad=True)
            for j in range(N_samples):
                K = Ks[j]
                if resample:
                    K = sampleMN(self.Ln_inv @ self.Qn, self.Ln_inv, self.sigeps)
                A[j, t, :, :] = K.T @ dphi_dxu[j, 0, :, :x_dim]
                B[j, t, :, :] = K.T @ dphi_dxu[j, 0, :, x_dim:]
                x_pred[j, t + 1, :] = x_pred[j, t, :] + np.squeeze(phi[j, :, :] @ K)

        return x_pred

    def reset_to_prior(self):
        self.Ln_inv = np.linalg.inv(self.L0)
        self.Qn = self.L0 @ self.K0

    def incorporate_transition(self, x, u, xp):
        x_inp = np.reshape(np.concatenate((x, u), axis=0), (1, 1, -1))
        phi = self.encode(x_inp)[0, :, :].T
        y = np.reshape(xp - x, (-1, 1))

        phiLamphi = (phi.T @ self.Ln_inv @ phi)[0, 0]
        LninvPhi = self.Ln_inv @ phi
        self.Ln_inv = self.Ln_inv - 1. / (1. + phiLamphi) * LninvPhi @ LninvPhi.T
        self.Qn = phi @ y.T + self.Qn

    def incorporate_batch(self, X, Y):
        Phi = self.encode(X)
        Kn, Ln_inv = blr_update_np(self.K0, self.L0, Phi[0, :, :], Y[0, :, :])
        self.Ln_inv = Ln_inv
        self.Kn = Kn
        self.Qn = np.linalg.inv(self.Ln_inv) @ self.Kn
