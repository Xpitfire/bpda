import torch
import numpy as np
import torch.nn.functional as F
from misc.layers import predict_from_logits
from misc.losses import F_cmd_loss, F_mmd_loss


def simple_MEAN(config, s_class_logits, s_class_target):
    """Computes the accuracies for source and target data"""

    s_class_preds = predict_from_logits(s_class_logits)
    acc = torch.mean((s_class_preds == s_class_target).float()).item()

    return {'acc': acc}


def accuracy(config, output, s_class_target, t_class_target):
    """Computes the accuracies for source and target data"""
    class_logits, domain_logits = output

    if "source_only" in config and config.source_only:
        s_class_logits = class_logits
        s_class_preds = predict_from_logits(s_class_logits)
        s_class_acc = torch.mean((s_class_preds == s_class_target).float()).item()
        t_class_acc = 0

        # total accuracy
        acc = np.mean([s_class_acc, t_class_acc])

    else:
        batchsplit = class_logits.shape[0] // 2
        s_class_logits = class_logits[:batchsplit, ...]
        t_class_logits = class_logits[batchsplit:, ...]

        s_class_preds = predict_from_logits(s_class_logits)
        t_class_preds = predict_from_logits(t_class_logits)

        s_class_acc = torch.mean((s_class_preds == s_class_target).float()).item()
        t_class_acc = torch.mean((t_class_preds == t_class_target).float()).item()

        # total accuracy
        acc = np.mean([s_class_acc, t_class_acc])

    return {'acc': acc, 's_class_acc': s_class_acc, 't_class_acc': t_class_acc}


def proxy_a_accuracy(config, output, s_class_target, t_class_target):
    """Computes the accuracies for source and target and domain data"""
    domain_logits, _ = output
    batchsize = domain_logits.shape[0] // 2

    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_domain_preds = predict_from_logits(s_domain_logits)
    t_domain_preds = predict_from_logits(t_domain_logits)

    s_domain_target = torch.zeros(batchsize).long().to(config.device)
    t_domain_target = torch.ones(batchsize).long().to(config.device)

    s_domain_acc = torch.mean((s_domain_preds == s_domain_target).float()).item()
    t_domain_acc = torch.mean((t_domain_preds == t_domain_target).float()).item()

    # total accuracy
    acc = np.mean([s_domain_acc, t_domain_acc])
    return {'acc': acc, 's_domain_acc': s_domain_acc, 't_domain_acc': t_domain_acc}


def domain_adversarial_accuracy(config, output, s_class_target, t_class_target):
    """Computes the accuracies for source and target and domain data"""
    class_logits, domain_logits = output
    batchsize = class_logits.shape[0] // 2

    s_class_logits = class_logits[:batchsize, ...]
    t_class_logits = class_logits[batchsize:, ...]
    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_class_preds = predict_from_logits(s_class_logits)
    t_class_preds = predict_from_logits(t_class_logits)
    s_domain_preds = predict_from_logits(s_domain_logits)
    t_domain_preds = predict_from_logits(t_domain_logits)

    s_domain_target = torch.zeros(batchsize).long().to(config.device)
    t_domain_target = torch.ones(batchsize).long().to(config.device)

    s_class_acc = torch.mean((s_class_preds == s_class_target).float()).item()
    t_class_acc = torch.mean((t_class_preds == t_class_target).float()).item()
    s_domain_acc = torch.mean((s_domain_preds == s_domain_target).float()).item()
    t_domain_acc = torch.mean((t_domain_preds == t_domain_target).float()).item()

    # total accuracy
    acc = np.mean([s_class_acc, t_class_acc])
    return {'acc': acc, 's_class_acc': s_class_acc, 't_class_acc': t_class_acc, 's_domain_acc': s_domain_acc,
            't_domain_acc': t_domain_acc}


def iwv_accuracy(config, output, s_class_target, t_class_target):
    """Computes the accuracies for source and target and domain data"""
    domain_logits, _ = output
    batchsize = domain_logits.shape[0] // 2

    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_domain_preds = predict_from_logits(s_domain_logits)
    t_domain_preds = predict_from_logits(t_domain_logits)

    s_domain_target = torch.zeros(batchsize).long().to(config.device)
    t_domain_target = torch.ones(batchsize).long().to(config.device)

    s_domain_acc = torch.mean((s_domain_preds == s_domain_target).float()).item()
    t_domain_acc = torch.mean((t_domain_preds == t_domain_target).float()).item()

    # total accuracy
    acc = np.mean([s_domain_acc, t_domain_acc])
    return {'acc': acc, 's_domain_acc': s_domain_acc, 't_domain_acc': t_domain_acc}


def cmd(config, output, s_class_target, t_class_target):
    class_logits, domain_logits = output
    batchsize = class_logits.shape[0] // 2

    s_class_logits = class_logits[:batchsize, ...]
    t_class_logits = class_logits[batchsize:, ...]
    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_class_preds = predict_from_logits(s_class_logits)
    t_class_preds = predict_from_logits(t_class_logits)

    s_class_acc = torch.mean((s_class_preds == s_class_target).float()).item()
    t_class_acc = torch.mean((t_class_preds == t_class_target).float()).item()

    # we don't need gradients for eval measures
    with torch.no_grad():
        cmd_distance = F_cmd_loss(s_domain_logits, t_domain_logits, config.trainer.cmd.moments).item()

    acc = np.mean([s_class_acc, t_class_acc])
    return {'acc': acc, 's_class_acc': s_class_acc, 't_class_acc': t_class_acc, 'da_metric': cmd_distance}


def mmd(config, output, s_class_target, t_class_target):
    class_logits, domain_logits = output
    batchsize = class_logits.shape[0] // 2

    s_class_logits = class_logits[:batchsize, ...]
    t_class_logits = class_logits[batchsize:, ...]
    s_domain_logits = domain_logits[:batchsize, ...]
    t_domain_logits = domain_logits[batchsize:, ...]

    s_class_preds = predict_from_logits(s_class_logits)
    t_class_preds = predict_from_logits(t_class_logits)

    s_class_acc = torch.mean((s_class_preds == s_class_target).float()).item()
    t_class_acc = torch.mean((t_class_preds == t_class_target).float()).item()

    # we don't need gradients for eval measures
    with torch.no_grad():
        mmd_distance = F_mmd_loss(s_domain_logits, t_domain_logits, config.trainer.mmd.sigma).item()

    acc = np.mean([s_class_acc, t_class_acc])
    return {'acc': acc, 's_class_acc': s_class_acc, 't_class_acc': t_class_acc, 'da_metric': mmd_distance}


def rho_func(output, y):
    max_f_x_y_prim = F.softmax(output.clone(), dim=0)
    f_x_y = F.softmax(output.clone(), dim=0)[y]
    max_f_x_y_prim[y] = -1
    max_f_x_y_prim = max_f_x_y_prim.max()

    if torch.isnan(f_x_y):
        print('NaN in f_x_y! {}'.format(f_x_y))

    if torch.isnan(max_f_x_y_prim):
        print('NaN in max_f_x_y_prim! {}'.format(max_f_x_y_prim))

    return 0.5 * (f_x_y - max_f_x_y_prim)


def phi_func(x, rho_param):
    if rho_param <= x:
        return 0
    elif 0 <= x and x <= rho_param:
        return 1 - (x / rho_param)
    elif x <= 0:
        return 1.
    else:
        print('illegal input! x:{} rho:{}'.format(x, rho_param))
        assert False


def disparity_loop(f_prime_output, f_output, rho_param):
    """

    :param f_prime_output: output of the last layer of the mdd classifier
    :param f_output: output of the last layer of the main classifier (source-only) which has been trained with DA.
    :param rho_param: a parameter that can be calculated from gamma using calc_rho_param
    :return:
    """
    batch_size = f_prime_output.shape[0]

    sum_disparity = 0
    for i in range(batch_size):
        y = f_output[i].argmax()
        rhp_func_out = rho_func(f_prime_output[i], y).detach().cpu().numpy()
        outputs = phi_func(rhp_func_out, rho_param=rho_param)
        sum_disparity += outputs

    return sum_disparity / batch_size


def calc_rho_param(gamma):
    """
    from the paper: gamma=exp(rho)
    :param gamma:
    :return: rho param
    """
    return np.log(gamma)


def mdd_distance(config, output, s_class_target, t_class_target):
    """
    Computes the loss required to train the MDD model
    :param config:
    :param output:
           source_only_clf_output: concatenated output of the last layer from source and target (validation set) using the source-only head's output.
            (the main classifier that was trained with DA, refered to as f in the paper.)
           mdd_clf_output: concatenated output of the last layer from source and target (validation set) using the mdd classifier model head's output.
           This is refered to as f' in the paper.
    :param gamma: the weight in eq. 30 of MDD paper.
    :return: losses
    """
    source_only_clf_output, mdd_clf_output = output

    batchsize = source_only_clf_output.shape[0] // 2
    s_source_only_clf_output = source_only_clf_output[:batchsize, ...]
    t_source_only_clf_output = source_only_clf_output[batchsize:, ...]

    s_mdd_clf_output = mdd_clf_output[:batchsize, ...]
    t_mdd_clf_output = mdd_clf_output[batchsize:, ...]

    s_f_prime = s_mdd_clf_output
    t_f_prime = t_mdd_clf_output

    s_f = s_source_only_clf_output
    t_f = t_source_only_clf_output

    mdd_dists = {}
    s_disparities = {}
    t_disparities = {}
    # compute multiple mdd distances with different gamma values
    for i in range(9):
        gamma = (i+2)*0.5
        rho_param = calc_rho_param(gamma)
        t_disparity = disparity_loop(t_f_prime, t_f, rho_param)
        if np.isnan(t_disparity):
            print('NaN in t_disparity! {}'.format(t_disparity))

        s_disparity = disparity_loop(s_f_prime, s_f, rho_param)
        if np.isnan(s_disparity):
            print('NaN in s_disparity! {}'.format(s_disparity))

        t_disparities[f't_disparity_gamma-{gamma}'] = t_disparity
        s_disparities[f's_disparity_gamma-{gamma}'] = s_disparity
        mdd_distance = t_disparity - s_disparity
        mdd_dists[f'mdd-dist_gamma-{gamma}'] = mdd_distance

    # metrics for statistics
    s_pseudo_target = s_source_only_clf_output.argmax(axis=1)
    t_pseudo_target = t_source_only_clf_output.argmax(axis=1)


    # pseudo label accuracy of the MDD classifier
    s_class_pseudo_mdd_acc = torch.mean((s_mdd_clf_output.argmax(dim=1) == s_pseudo_target).float()).item()
    t_class_pseudo_mdd_acc = torch.mean(
        ((1. - F.softmax(t_mdd_clf_output, dim=1)).argmax(dim=1) == t_pseudo_target).float()).item()

    # classification accuracy of the source-only classifier
    s_class_acc = torch.mean((s_source_only_clf_output.argmax(dim=1) == s_class_target).float()).item()
    t_class_acc = torch.mean(
        ((1. - F.softmax(t_source_only_clf_output, dim=1)).argmax(dim=1) == t_class_target).float()).item()


    acc = np.mean([s_class_acc, t_class_acc])
    return {'acc': acc,
            's_mdd_cls_acc': s_class_acc,'t_mdd_cls_acc': t_class_acc,
            's_class_pseudo_mdd_acc': s_class_pseudo_mdd_acc, 't_class_pseudo_mdd_acc': t_class_pseudo_mdd_acc,
            **t_disparities, **s_disparities, **mdd_dists}
