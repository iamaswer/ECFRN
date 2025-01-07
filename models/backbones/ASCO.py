import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import torch.optim as optim
from trainers.eval import meta_test
import numpy as np

class ASCO(nn.Module):
    def __init__(self):
        super(ASCO, self).__init__()
        self.linear1 = nn.Linear(2, 2)  # Input: [C, K], Output: intermediate
        self.linear2 = nn.Linear(2, 1)  # Input: intermediate, Output: phi

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)

    def predict_phi(self, C, K):
        # Predict phi given C and K
        with torch.no_grad():
            input_tensor = torch.tensor([[C, K]], dtype=torch.float32)
            phi_pred = self.forward(input_tensor).item()
        return int(np.ceil(max(1, min(phi_pred, C))))  # Ensure 1 <= phi' <= C

class Train_ASCO:
    def __init__(self, pretrained_model=None, model = None, data_path=None, save_path = None, learning_rate=0.01, max_epochs=300, decay_epochs=100):
        self.data_path = data_path
        self.save_path = save_path
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.decay_epochs = decay_epochs
        self.model = model  # Embed the linear model within ASCO
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def prepare_experimental_dataset(self, C_values, K_values, pre, transform_type):
        Dexp = []  # Experimental dataset
        for C in C_values:
            for K in K_values:
                phi_mean = []
                for phi in range(1, C + 1):
                    mean, interval = meta_test(
                        data_path=self.data_path,
                        model=self.pretrained_model,
                        way=C,
                        shot=K,
                        pre=pre,
                        phi=phi,
                        transform_type=transform_type,
                        trial=10
                    )
                    phi_mean.append((phi, mean))
                best_phi, max_mean = max(phi_mean, key=lambda x: x[1])
                print(C, K, best_phi)
                Dexp.append([C, K, best_phi])
        self.Dexp = torch.tensor(Dexp, dtype=torch.float32)  # Shape: [T, 3]

    def train(self):
        if not hasattr(self, 'Dexp'):
            raise AttributeError("Experimental dataset not prepared. Call prepare_experimental_dataset first.")
        
        # Extract features (C, K) and target (phi)
        X = self.Dexp[:, :2]  # Features: C and K
        y = self.Dexp[:, 2].unsqueeze(1)  # Target: phi

        for epoch in range(1, self.max_epochs + 1):
            # Reduce learning rate after decay_epochs
            if epoch % self.decay_epochs == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1

            # Forward pass
            predictions = self.model(X)
            loss = self.criterion(predictions, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss for monitoring
            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.max_epochs}, Loss: {loss.item():.6f}")
        torch.save(self.model.state_dict(),self.save_path)

