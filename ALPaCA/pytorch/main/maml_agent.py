import numpy as np
import torch
import torch.optim as optim
from maml.maml import MAML
import yaml
import matplotlib.pyplot as plt


class MAMLAgent:
    def __init__(self, config, device=None, exp_string="maml_test"):
        self.config = config
        self.device = device if device is not None else torch.device('cpu')
        self.logdir = "summaries"
        self.exp_string = exp_string
        self.test_num_updates = 5

        # Default settings for sinusoid regression
        self.update_batch_size = 10
        self.meta_batch_size = 25
        self.meta_lr = 0.001
        self.update_lr = 1e-3
        self.num_updates = 1

    def construct_model(self, load_model=True):
        self.model = MAML(
            self.config['x_dim'], self.config['y_dim'],
            self.test_num_updates,
            update_lr=self.update_lr,
            meta_lr=self.meta_lr
        )
        self.model.construct_model()
        self.model.to(self.device)

        if load_model:
            import os
            model_dir = os.path.join(self.logdir, self.exp_string)
            if os.path.exists(model_dir):
                # Try to find latest checkpoint
                checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                if checkpoints:
                    latest = sorted(checkpoints)[-1]
                    model_file = os.path.join(model_dir, latest)
                    print("Restoring model weights from " + model_file)
                    self.model.load_state_dict(
                        torch.load(model_file, map_location=self.device, weights_only=True)
                    )

    def train(self, dataset, metatrain_iterations, pretrain_iterations=0):
        PRINT_INTERVAL = 100
        SAVE_INTERVAL = 1000
        update_batch_size = self.update_batch_size
        meta_batch_size = self.meta_batch_size

        print('Done initializing, starting training.')
        prelosses, postlosses = [], []
        num_updates = max(self.num_updates, self.test_num_updates)

        for itr in range(pretrain_iterations + metatrain_iterations):
            X, Y = dataset.sample(meta_batch_size, update_batch_size * 2)

            inputa = torch.tensor(X[:, :update_batch_size, :], dtype=torch.float32, device=self.device)
            labela = torch.tensor(Y[:, :update_batch_size, :], dtype=torch.float32, device=self.device)
            inputb = torch.tensor(X[:, update_batch_size:, :], dtype=torch.float32, device=self.device)
            labelb = torch.tensor(Y[:, update_batch_size:, :], dtype=torch.float32, device=self.device)

            self.model.meta_optimizer.zero_grad()
            total_loss1, total_losses2, outputbs = self.model(
                inputa, inputb, labela, labelb, num_updates
            )

            if itr < pretrain_iterations:
                total_loss1.backward()
            else:
                total_losses2[num_updates - 1].backward()

            self.model.meta_optimizer.step()

            prelosses.append(total_loss1.item())
            postlosses.append(total_losses2[-1].item())

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                if itr < pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - pretrain_iterations)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []

            if (itr != 0) and itr % SAVE_INTERVAL == 0:
                import os
                save_dir = os.path.join(self.logdir, self.exp_string)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, 'model' + str(itr) + '.pt'))

    def test(self, ux, uy, x, num_updates=1):
        self.model.eval()
        with torch.no_grad():
            ux_t = torch.tensor(ux, dtype=torch.float32, device=self.device)
            uy_t = torch.tensor(uy, dtype=torch.float32, device=self.device)
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

            # For testing, we need to do inner loop with grad enabled
        # Re-enable grad for inner loop
        ux_t = torch.tensor(ux, dtype=torch.float32, device=self.device)
        uy_t = torch.tensor(uy, dtype=torch.float32, device=self.device)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

        meta_batch_size = ux_t.shape[0]
        all_outputs = [[] for _ in range(num_updates)]

        with torch.no_grad():
            from collections import OrderedDict
            for i in range(meta_batch_size):
                weights = OrderedDict(self.model.weights.items())

                # Inner loop updates using support set
                for step in range(num_updates):
                    # Need grad for inner loop
                    pass

        # Simpler approach: use the model's task_metalearn but without meta gradients
        all_outputbs = []
        for i in range(meta_batch_size):
            weights = dict(self.model.weights.items())
            # Detach weights for inner loop
            fast_weights = {k: v.clone().detach().requires_grad_(True) for k, v in weights.items()}

            for step in range(num_updates):
                output_a = self.model.forward_fc(ux_t[i], fast_weights)
                loss_a = self.model.loss_func(output_a, uy_t[i])
                grads = torch.autograd.grad(loss_a, list(fast_weights.values()), create_graph=False)
                fast_weights = {
                    name: param - self.model.update_lr * grad
                    for (name, param), grad in zip(fast_weights.items(), grads)
                }
                # Re-enable grad on new fast_weights
                fast_weights = {k: v.detach().requires_grad_(True) for k, v in fast_weights.items()}

            with torch.no_grad():
                output_b = self.model.forward_fc(x_t[i], fast_weights)
                all_outputbs.append(output_b)

        y = torch.stack(all_outputbs).cpu().numpy()
        return y, None


class MAMLDynamics(MAMLAgent):
    def __init__(self, config, device=None, exp_string="maml_test"):
        super(MAMLDynamics, self).__init__(config, device, exp_string)
        self.ux = []
        self.uy = []

    def sample_rollout(self, x0, actions):
        T, a_dim = actions.shape
        mult_sample = False
        if x0.ndim == 1:
            N_samples = 1
            x_dim = x0.shape[0]
            x0 = np.expand_dims(x0, axis=1)
        elif x0.ndim == 2:
            mult_sample = True
            N_samples = x0.shape[0]
            x_dim = x0.shape[1]

        actions = np.tile(np.expand_dims(actions, axis=0), (N_samples, 1, 1))

        x_pred = np.zeros((N_samples, T + 1, x_dim))
        x_pred[:, 0, :] = x0

        if len(self.ux) > 0:
            UX = np.concatenate(self.ux, axis=1)
            UY = np.concatenate(self.uy, axis=1)
        else:
            UX = np.zeros([1, 0, self.config['x_dim']])
            UY = np.zeros([1, 0, self.config['y_dim']])

        for t in range(0, T):
            x_inp = np.concatenate((x_pred[0:1, t:t + 1, :], actions[0:1, t:t + 1, :]), axis=2)
            y, s = self.test(UX, UY, x_inp, num_updates=5)
            x_pred[:, t + 1, :] = y + x_pred[:, t, :]

        if mult_sample:
            return x_pred[:, 1:, :]
        else:
            return x_pred[0, 1:, :]

    def reset_to_prior(self):
        self.ux = []
        self.uy = []

    def incorporate_transition(self, x, u, xp):
        x_inp = np.reshape(np.concatenate((x, u), axis=0), (1, 1, -1))
        y = np.reshape(xp - x, (1, 1, -1))

        self.ux.append(x_inp)
        self.uy.append(y)