import torch
import torch.nn.functional as F
from losses import F_mmd_loss, F_cmd_loss


def simple_CE(config, s_class_logits, s_class_target):
    loss = F.cross_entropy(s_class_logits, s_class_target)
    # collect the loss that will be used for optimization
    return loss, {'loss': loss.item()}


def source_only(config, output, s_class_target, t_class_target):
    class_logits, domain_logits = output

    if "source_only" in config and config.source_only:
        s_class_logits = class_logits
        s_class_loss = F.cross_entropy(s_class_logits, s_class_target)
        t_class_loss = torch.zeros(1).to(class_logits.device)
    else:
        batchsplit = class_logits.shape[0] // 2
        s_class_logits = class_logits[:batchsplit, ...]
        t_class_logits = class_logits[batchsplit:, ...]
        s_class_loss = F.cross_entropy(s_class_logits, s_class_target)
        t_class_loss = F.cross_entropy(t_class_logits, t_class_target)  # not used for optimization!

    # collect the loss that will be used for optimization
    loss = s_class_loss
    return loss, {'loss': loss.item(), 's_class_loss': s_class_loss.item(), 't_class_loss': t_class_loss.item()}


def proxy_a_classifier_ce(config, output, s_class_target, t_class_target):
    domain_logits, _ = output
    batchsize = domain_logits.shape[0] // 2

    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_domain_target = torch.zeros(batchsize).long().to(config.device)
    t_domain_target = torch.ones(batchsize).long().to(config.device)

    s_domain_loss = F.nll_loss(F.log_softmax(s_domain_logits, dim=-1), s_domain_target)
    t_domain_loss = F.nll_loss(F.log_softmax(t_domain_logits, dim=-1), t_domain_target)

    # collect the loss that will be used for optimization
    loss = s_domain_loss + t_domain_loss
    return loss, {'loss': loss.item(), 's_class_loss': s_domain_loss.item(), 't_class_loss': t_domain_loss.item()}


def mdd_classifier_ce(config, output, s_class_target, t_class_target):
    """
    Computes the loss required to train the MDD model
    :param config:
    :param output: tuple of logits
           source_only_clf_output: concatenated output of the last layer from source and target (training set) using the source-only head's output  (the main classifier that was trained with DA, refered to as f in the paper.)
           mdd_clf_output: concatenated output of the last layer from source and target (training set) using the mdd classifier model head's output. This is refered to as f' in the paper.
    :param s_class_target: source class labels
    :param t_class_target: target class labels (should not be used for training)
    :param gamma: the weight in eq. 30 of MDD paper.
    :return: losses
    """
    source_only_clf_output, mdd_clf_output = output

    batchsize = source_only_clf_output.shape[0] // 2
    s_source_only_clf_output = source_only_clf_output[:batchsize, ...]
    t_source_only_clf_output = source_only_clf_output[batchsize:, ...]

    # create pseudo labels out of logits
    s_pseudo_target = s_source_only_clf_output.argmax(axis=1)
    t_pseudo_target = t_source_only_clf_output.argmax(axis=1)

    batchsize = mdd_clf_output.shape[0] // 2

    s_mdd_clf_output = mdd_clf_output[:batchsize, ...]
    t_mdd_clf_output = mdd_clf_output[batchsize:, ...]

    s_mdd_clf_loss = F.cross_entropy(s_mdd_clf_output, s_pseudo_target)
    t_mdd_clf_loss = F.nll_loss(torch.log(1 - F.softmax(t_mdd_clf_output, dim=1)), t_pseudo_target)


    if torch.isnan(s_mdd_clf_loss).any():
        print('NaN in s_mdd_clf_loss! {}'.format(s_mdd_clf_loss))
        assert False

    if torch.isnan(t_mdd_clf_loss).any():
        print('NaN in t_mdd_clf_loss! {}'.format(t_mdd_clf_loss))
        assert False

    # transfer loss we need for multi-class
    transfer_loss = config.approach.gamma_ * s_mdd_clf_loss + t_mdd_clf_loss

    return transfer_loss, {'loss': transfer_loss.item(),
                           'transfer_loss': transfer_loss.item(),
                           's_mdd_clf_loss': s_mdd_clf_loss.item(),
                           't_mdd_clf_loss': t_mdd_clf_loss.item()}


def domain_adversarial(config, output, s_class_target, t_class_target):
    """Computes cross entropy and adversarial loss based on the source and target data (without target labels).
    Based on the domain Domain Adversarial Neural Network paper https://arxiv.org/abs/1505.07818 
    """
    class_logits, domain_logits = output
    batchsize = class_logits.shape[0] // 2

    s_class_logits = class_logits[:batchsize, ...]
    t_class_logits = class_logits[batchsize:, ...]
    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_domain_target = torch.zeros(batchsize).long().to(config.device)
    t_domain_target = torch.ones(batchsize).long().to(config.device)

    s_class_loss = F.cross_entropy(s_class_logits, s_class_target)
    t_class_loss = F.cross_entropy(t_class_logits, t_class_target)  # not used for optimization!
    s_domain_loss = F.nll_loss(F.log_softmax(s_domain_logits, dim=-1), s_domain_target)
    t_domain_loss = F.nll_loss(F.log_softmax(t_domain_logits, dim=-1), t_domain_target)

    # collect the loss that will be used for optimization
    loss = s_class_loss + config.approach.lambda_ * (s_domain_loss + t_domain_loss)
    return loss, {'loss': loss.item(), 's_class_loss': s_class_loss.item(), 't_class_loss': t_class_loss.item(),
                  's_domain_loss': s_domain_loss.item(), 't_domain_loss': t_domain_loss.item()}


def iwv(config, output, s_class_target, t_class_target):
    """Computes cross entropy and adversarial loss based on the source and target data (without target labels).
    Based on the domain Domain Adversarial Neural Network paper https://arxiv.org/abs/1505.07818 
    """
    domain_logits, _ = output
    batchsize = domain_logits.shape[0] // 2

    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_domain_target = torch.zeros(batchsize).long().to(config.device)
    t_domain_target = torch.ones(batchsize).long().to(config.device)

    s_domain_loss = F.nll_loss(F.log_softmax(s_domain_logits, dim=-1), s_domain_target)
    t_domain_loss = F.nll_loss(F.log_softmax(t_domain_logits, dim=-1), t_domain_target)

    # collect the loss that will be used for optimization
    loss = s_domain_loss + t_domain_loss
    return loss, {'loss': loss.item(), 's_domain_loss': s_domain_loss.item(), 't_domain_loss': t_domain_loss.item()}


def cmd(config, output, s_class_target, t_class_target):
    class_logits, domain_logits = output
    batchsize = class_logits.shape[0] // 2

    s_class_logits = class_logits[:batchsize, ...]
    t_class_logits = class_logits[batchsize:, ...]
    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_class_loss = F.cross_entropy(s_class_logits, s_class_target)
    t_class_loss = F.cross_entropy(t_class_logits, t_class_target)  # not used for optimization!

    domain_loss = F_cmd_loss(s_domain_logits, t_domain_logits, config.trainer.cmd.moments)

    # collect the loss that will be used for optimization
    loss = s_class_loss + config.approach.lambda_ * domain_loss
    return loss, {'loss': loss.item(), 's_class_loss': s_class_loss.item(), 't_class_loss': t_class_loss.item(),
                  'da_loss': config.approach.lambda_ * domain_loss.item()}


def mmd(config, output, s_class_target, t_class_target):
    class_logits, domain_logits = output
    batchsize = class_logits.shape[0] // 2

    s_class_logits = class_logits[:batchsize, ...]
    t_class_logits = class_logits[batchsize:, ...]
    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_class_loss = F.cross_entropy(s_class_logits, s_class_target)
    t_class_loss = F.cross_entropy(t_class_logits, t_class_target)  # not used for optimization!

    domain_loss = F_mmd_loss(s_domain_logits, t_domain_logits, config.trainer.mmd.sigma)

    # collect the loss that will be used for optimization
    loss = s_class_loss + config.approach.lambda_ * domain_loss
    return loss, {'loss': loss.item(), 's_class_loss': s_class_loss.item(), 't_class_loss': t_class_loss.item(),
                  'da_loss': config.approach.lambda_ * domain_loss.item()}

