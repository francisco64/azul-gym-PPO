import torch
import torch.nn as nn
from torchsummary import summary

class FlattenExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
    def forward(self, x):
        return self.flatten(x)


class MlpExtractor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.Tanh()
        )
        self.policy_net = nn.Sequential()
        self.value_net = nn.Sequential()
        
    def forward(self, x):
        shared_out = self.shared_net(x)
        policy_out = self.policy_net(shared_out)
        value_out = self.value_net(shared_out)
        return policy_out, value_out


class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = FlattenExtractor()
        self.mlp_extractor = MlpExtractor(159)
        self.action_net = nn.Linear(150, 150)
        self.value_net = nn.Linear(150, 1)

    def forward(self, x):
        x = self.features_extractor(x)
        policy_out, value_out = self.mlp_extractor(x)
        action_out = self.action_net(policy_out)
        value_out = self.value_net(value_out)
        return action_out, value_out

# Instantiate the model
model = ActorCriticPolicy()

# Summary of the model
summary(model, (1, 159))
