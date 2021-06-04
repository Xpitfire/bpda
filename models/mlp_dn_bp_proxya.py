import torch.nn as nn
from misc.layers import activations
from models.backbone import set_gradients_enabled, register_domain_adaptation_features_extraction


class MLP(nn.Module):
    latent_features = {}
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.backbone = None
        # input layer projection
        self.in_layer = nn.Linear(config.backbone.out_units, config.model.MLP.n_hidden//2)
        self.in_norm = nn.BatchNorm1d(config.model.MLP.n_hidden//2)
        self.phi = activations[config.model.MLP.activation]()
        self.layers = nn.ModuleList()
        # create a variable list of hidden layers
        for i in range(config.model.MLP.n_layers):
            self.layers.append(nn.Linear(config.model.MLP.n_hidden//2, config.model.MLP.n_hidden//2))
            self.layers.append(nn.BatchNorm1d(config.model.MLP.n_hidden//2))
            self.layers.append(activations[config.model.MLP.activation]())
            self.layers.append(nn.Dropout(p=config.model.MLP.dropout))
        # output head for the class classifier
        self.out_class = nn.Sequential(
            nn.Linear(config.model.MLP.n_hidden//2, 2)
        )
    
    def set_backbone(self, backbone):
        backbone = backbone.to(self.config.device)
        backbone.eval()
        set_gradients_enabled(self.config, backbone)
        self.backbone = register_domain_adaptation_features_extraction(self.config, backbone)

    def forward(self, x, *args):
        h = self.backbone(x)
        h = self.in_layer(h)
        h = self.in_norm(h)
        h = self.phi(h)
        for layer in self.layers:
            h = layer(h)
        c = self.out_class(h)
        return c, h
