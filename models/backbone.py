import torch
import torchvision.models as models


def set_gradients_enabled(config, model):
    print('Enbable backbone gradients: ', config.backbone.trainable)
    for param in model.parameters():
        param.requires_grad = config.backbone.trainable


def register_domain_adaptation_features_extraction(config, model):
    print('Registered feature extraction hook')
    latent_features = {}
    def register_hook():
        def hook(model, input, output):
            latent_features[config.backbone.feature_layer] = output if config.backbone.trainable else output.detach()
        return hook
    getattr(model, config.backbone.feature_layer).register_forward_hook(register_hook())
    def features(x):
        if config.backbone.trainable:
            model(x)
        else:
            # don't compute gradients if not trainable
            with torch.no_grad():
                model(x)
        return latent_features[config.backbone.feature_layer]
    return features


def register_resnet_features_extraction(config, model):
    print('Registered feature extraction hook')
    latent_features = {}
    def register_hook():
        def hook(model, input, output):
            latent_features[config.backbone.feature_layer] = output if config.backbone.trainable else output.detach()
        return hook
    getattr(model, config.backbone.feature_layer).register_forward_hook(register_hook())
    def features(x):
        if config.backbone.trainable:
            model(x)
        else:
            # don't compute gradients if not trainable
            with torch.no_grad():
                model(x)
        return latent_features[config.backbone.feature_layer]
    return features


def register_alexnet_features_extraction(config, model):
    print('Registered feature extraction hook')
    latent_features = {}
    def register_hook():
        def hook(model, input, output):
            latent_features[config.backbone.feature_layer] = output if config.backbone.trainable else output.detach()
        return hook
    getattr(model, config.backbone.feature_layer)[config.backbone.feature_layer_idx].register_forward_hook(register_hook())
    def features(x):
        if config.backbone.trainable:
            model(x)
        else:
            # don't compute gradients if not trainable
            with torch.no_grad():
                model(x)
        return latent_features[config.backbone.feature_layer]
    return features


def get_backbone(config):
    model = None
    params = None
    num_features = None
    if config.backbone.model == "resnet18":
        print('Selected ResNet18 backbone')
        model = models.resnet18(pretrained=config.backbone.pretrained)
        if config.debug: print(model)
        model = model.to(config.device)
        params = model.parameters()
        num_features = list(getattr(model, config.backbone.feature_layer).parameters())[-1].shape[0]
        set_gradients_enabled(config, model)
        extractor = register_resnet_features_extraction(config, model)
    elif config.backbone.model == "alexnet":
        print('Selected AlexNet backbone')
        model = models.alexnet(pretrained=config.backbone.pretrained)
        if config.debug: print(model)
        model = model.to(config.device)
        params = model.parameters()
        num_features = list(getattr(model, config.backbone.feature_layer)[config.backbone.feature_layer_idx].parameters())[-1].shape[0]
        set_gradients_enabled(config, model)
        extractor = register_alexnet_features_extraction(config, model)
    else:
        raise NotImplementedError("Not Implemented!")
    return extractor, model, params, num_features
