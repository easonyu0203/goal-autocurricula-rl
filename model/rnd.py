import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

class RandomTargetNetwork(nn.Module):
    def __init__(self, input_shape, convfeat=32, rep_size=512):
        super(RandomTargetNetwork, self).__init__()
        c, h, w = input_shape

        # Using nn.Sequential for better readability
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, convfeat, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        
        # Calculate output size after the convolutional layers
        conv_out_size = self._get_conv_out((c, h, w))
        
        # Fully connected layer to generate feature representation
        self.fc = nn.Linear(conv_out_size, rep_size)

        # Apply orthogonal initialization
        self._initialize_weights()

    def _get_conv_out(self, shape):
        """Helper to compute the size after convolution layers."""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

    def _initialize_weights(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()

class PredictorNetwork(nn.Module):
    def __init__(self, input_shape, convfeat=32, rep_size=512):
        super(PredictorNetwork, self).__init__()
        c, h, w = input_shape

        # Using nn.Sequential for better readability
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, convfeat, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        # Calculate output size after convolutions
        conv_out_size = self._get_conv_out((c, h, w))

        # Fully connected layers for prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, rep_size)
        )

        # Apply orthogonal initialization
        self._initialize_weights()

    def _get_conv_out(self, shape):
        """Helper to compute size of tensor after convolution layers."""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_layers(x)

    def _initialize_weights(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                if p.bias is not None:
                    p.bias.data.zero_()

class RNDModel(nn.Module):
    def __init__(self, input_shape, convfeat=32, rep_size=512):
        super(RNDModel, self).__init__()
        
        # Initialize the target and predictor networks
        self.target_network = RandomTargetNetwork(input_shape, convfeat, rep_size)
        self.predictor_network = PredictorNetwork(input_shape, convfeat, rep_size)
        
        # Freeze the target network to prevent gradient updates
        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of observations (shape: [batch_size, channels, height, width]).

        Returns:
            predict_feature (torch.Tensor): Feature predicted by the predictor network.
            target_feature (torch.Tensor): Target feature from the target network.
        """
        # Get features from both networks
        target_feature = self.target_network(x)
        predict_feature = self.predictor_network(x)

        return predict_feature, target_feature
    

class DRNDModel(nn.Module):
    def __init__(self, input_shape, convfeat=32, rep_size=512, num_targets=10):
        super(DRNDModel, self).__init__()
        
        # Initialize multiple target networks and the predictor network
        self.target_networks = nn.ModuleList([RandomTargetNetwork(input_shape, convfeat, rep_size) for _ in range(num_targets)])
        self.predictor_network = PredictorNetwork(input_shape, convfeat, rep_size)
        
        # Freeze the target networks to prevent gradient updates
        for target_network in self.target_networks:
            for param in target_network.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of observations (shape: [batch_size, channels, height, width]).

        Returns:
            predict_feature (torch.Tensor): Feature predicted by the predictor network.
            target_feature (torch.Tensor): Mean feature from all target networks.
        """
        # Get features from all target networks and compute the mean
        target_features = [target_network(x) for target_network in self.target_networks]
        target_feature = torch.mean(torch.stack(target_features), dim=0)
        
        # Get the feature from the predictor network
        predict_feature = self.predictor_network(x)

        return predict_feature, target_feature
    

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count