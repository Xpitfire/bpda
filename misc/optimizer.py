import torch
import torch.optim as optim
import torch


def Adam(config, model):
    """Adam shortcut configuration to switch the optimizer for training via config"""
    scheduler = None
    optimizer = optim.Adam(model.parameters(),
                           lr=config.trainer.Adam.lr,
                           betas=(config.trainer.Adam.beta1, config.trainer.Adam.beta2),
                           weight_decay=config.trainer.Adam.weight_decay)
    print('Selected Adam optimizer!')
    if config.trainer.Adam.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config.trainer.Adam.MultiStepLR.milestones,
                                                         gamma=config.trainer.Adam.MultiStepLR.gamma)
        print('Selected Adam with MultiStepLR scheduler optimizer!')
    elif config.trainer.Adam.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=config.trainer.Adam.ReduceLROnPlateau.mode,
                                                               factor=config.trainer.Adam.ReduceLROnPlateau.factor,
                                                               patience=config.trainer.Adam.ReduceLROnPlateau.patience,
                                                               min_lr=config.trainer.Adam.ReduceLROnPlateau.min_lr)
    elif config.trainer.Adam.scheduler == "LambdaLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=config.approach.lr_lambda(
            mu0=config.trainer.Adam.LambdaLR.mu0,
            alpha=config.trainer.Adam.LambdaLR.alpha,
            beta=config.trainer.Adam.LambdaLR.beta))
    elif config.trainer.Adam.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.trainer.Adam.CosineAnnealingLR.t_max,
                                                               eta_min=config.trainer.Adam.CosineAnnealingLR.eta_min)
    return optimizer, scheduler


def SGD(config, model):
    """SGD shortcut configuration to switch the optimizer for training via config"""
    scheduler = None
    optimizer = optim.SGD(model.parameters(),
                          lr=config.trainer.SGD.lr,
                          momentum=config.trainer.SGD.momentum,
                          weight_decay=config.trainer.SGD.weight_decay)
    print('Selected SGD optimizer!')
    if config.trainer.SGD.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.trainer.SGD.StepLR.step_size,
                                                    gamma=config.trainer.SGD.StepLR.gamma)
        print('Selected SGD with StepLR scheduler optimizer!')
    elif config.trainer.SGD.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=config.trainer.SGD.MultiStepLR.milestones,
                                                         gamma=config.trainer.SGD.MultiStepLR.gamma)
        print('Selected SGD with MultiStepLR scheduler optimizer!')
    elif config.trainer.SGD.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=config.trainer.SGD.ReduceLROnPlateau.mode,
                                                               factor=config.trainer.SGD.ReduceLROnPlateau.factor,
                                                               patience=config.trainer.SGD.ReduceLROnPlateau.patience,
                                                               min_lr=config.trainer.SGD.ReduceLROnPlateau.min_lr)
    elif config.trainer.SGD.scheduler == "LambdaLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=config.approach.lr_lambda(
            mu0=config.trainer.SGD.LambdaLR.mu0,
            alpha=config.trainer.SGD.LambdaLR.alpha,
            beta=config.trainer.SGD.LambdaLR.beta))
    elif config.trainer.SGD.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.trainer.SGD.CosineAnnealingLR.t_max,
                                                               eta_min=config.trainer.SGD.CosineAnnealingLR.eta_min)
    return optimizer, scheduler
