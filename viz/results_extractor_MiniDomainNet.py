import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def extract_risks(lamb_list, seeds, metric, domain, s_cls_err=None, t_cls_err=None, d_proxya=None, bp_proxya=None, d_mdd=None, bp_mdd=None, iwv_alg=None,  dev_alg=None):
    dev = np.mean(dev_alg, axis=1)
    iwv = np.mean(iwv_alg, axis=1)
    min_idx_iwcv = np.around(np.where(dev == np.nanmin(dev))[0], decimals=3)
    min_idx_dev = np.around(np.where(iwv == np.nanmin(iwv))[0], decimals=3)

    print(f"results {metric} {domain} all seeds")
    test_mean = np.around(np.mean(t_cls_err, axis=1), decimals=3)
    test_std = np.around(np.std(t_cls_err, axis=1), decimals=3)
    print(f"\tSource only \tmean_val: {test_mean[0]} mean_std: {test_std[0]}")
    print(
        f"\tIWCV_orig min indx: {min_idx_iwcv} \tmean_val: {[test_mean[i] for i in min_idx_iwcv]} \tmean_std: {[test_std[i] for i in min_idx_iwcv]} \tweight: {[lamb_list[i] for i in min_idx_iwcv]}")
    print(
        f"\tDEV_orig min indx {min_idx_dev}  \tmean_val: {[test_mean[i] for i in min_idx_dev]} \tmean_std: {[test_std[i] for i in min_idx_dev]} \tweight: {[lamb_list[i] for i in min_idx_dev]}")
    print(
        f"\tBP MDD minimum \tmean_val: {test_mean[np.where(bp_mdd == np.nanmin(bp_mdd))[0][-1] + 1]} \tmean_std: {test_std[np.where(bp_mdd == np.nanmin(bp_mdd))[0][-1] + 1]} \tweight: {lamb_list[np.where(bp_mdd == np.nanmin(bp_mdd))[0][-1] + 1]}")

    print(
        f"\tTarget best \tmean_val: {np.nanmin(test_mean)} \tmean_std: {test_std[np.where(test_mean == np.nanmin(test_mean))[0][0]]} \tweight: {lamb_list[np.where(test_mean == np.nanmin(test_mean))[0][0]]}")

    risk_dict = {"s_cls_err": s_cls_err,
                 "t_cls_err": t_cls_err}


    plot_risks(risk_dict, lamb_list, 'eval', domain, metric,  d_proxya, bp_proxya, d_mdd, bp_mdd, iwv, dev)


def plot_risks(risk_dict, lambda_list, key, domain, metric,  adist=None, vote_adist=None, mmd=None, vote_mdd=None, iwcv=None, dev=None):
    ww = np.arange(len(lambda_list))
    mpl.rcParams['figure.dpi'] = 300
    sns.set_style("whitegrid")

    sns.color_palette("colorblind", 8)
    fig, ax = plt.subplots()
    ax.set_xticks(ww)
    ax.set_yticks([i for i in np.arange(-0.2, 1.1, 0.05)])
    ax.set_ylim(-0.2, 1.1)
    for k, val in risk_dict.items():
        mean_ = np.mean(val, axis=1)
        var_ = np.std(val, axis=1)
        if k=='cmd' or k=='mmd':
            mean_ = (mean_ - min(mean_)) / (max(mean_) - min(mean_))
            var_ = (var_ - min(var_)) / (max(var_) - min(var_))
        ax.plot(ww, mean_, '.-', label=k)
        ax.fill_between(ww, mean_ - var_, mean_ + var_, alpha=0.2)

    if adist is not None:
        mean_ = np.mean(adist, axis=1)
        var_ = np.std(adist, axis=1)
        ax.plot(ww, mean_, '.-', label='A-dist')
        ax.fill_between(ww, mean_ - var_, mean_ + var_, alpha=0.2)

    if vote_adist is not None:
        ax.plot(ww[1::], vote_adist, '.-', label="A-dist Votes")


    if mmd is not None:
        mean_ = np.mean(mmd, axis=1)
        var_ = np.std(mmd, axis=1)
        ax.plot(ww, mean_, '.-', label='MDD')
        ax.fill_between(ww, mean_ - var_, mean_ + var_, alpha=0.2)

    if vote_mdd is not None:
        ax.plot(ww[1::], vote_mdd, '.-', label="MDD Votes")


    if iwcv is not None:
        mean_ = (iwcv - min(iwcv)) / (max(iwcv) - min(iwcv))
        var_ = (iwcv - min(iwcv)) / (max(iwcv) - min(iwcv))

        ax.plot(ww, mean_, '.-', label="IWCV")
        ax.fill_between(ww, mean_ - var_, mean_ + var_, alpha=0.2)


    if dev is not None:
        mean_ = (dev - min(dev)) / (max(dev) - min(dev))
        var_ = (dev - min(dev)) / (max(dev) - min(dev))
        ax.plot(ww, mean_, '.-', label="DEV")
        ax.fill_between(ww, mean_ - var_, mean_ + var_, alpha=0.2)


    ax.set_xticklabels(lambda_list)
    ax.set_xlabel('weight')
    ax.set_ylabel('value')
    ax.set_title(
        '{} {} {}'.format(metric, domain, key),
        fontsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    plt.legend(fontsize=6)
    sns.despine()
    plt.show()
    plt.close()


def load_results_allseeds(base_folder, file_name, method, dataset, exp_key, seeds=None):
    """Loads seed results from a file
    param: file: filename of experimental resuts to be loaded
    param: exp_key: experiment key name e.g. 'clipart-infograph-0'. This may depend on the result file if it includes lambda values attached or not
    param: seed: experiment seed. the seed can be replaced by a wildcard 'any' if there is only one seed available
    param: run_key: field access to be returned
    param: idx: return the idx log of file if available
    """
    dirs = os.listdir(base_folder)

    if seeds == None:
        all_domain_dirs = [dr for dr in dirs if f'{method}_{dataset}' in dr]
    else:
        all_domain_dirs = [dr for dr in dirs for seed in seeds if f'{method}_{dataset}_{seed}' in dr]

    accum_res = []
    for domain_dir in all_domain_dirs:
        res = np.load(os.path.join(base_folder, os.path.join(domain_dir, file_name)), allow_pickle=True)
        res_ = np.array(res['arr_0']).item()
        seed = list(res_[exp_key].keys())[0]
        val = res_[exp_key][seed]
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        accum_res.append(val)

    return accum_res


def flatten(batch_list):
    return np.concatenate(batch_list, axis=0)


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x[:] - np.max(x, axis=dim)[:, None])
    return e_x / e_x.sum(axis=dim)[:, None]


def predict(probs, dim=1):
    return np.argmax(probs, axis=dim)


def get_weight(s_val_preds, s_N, t_N):
    probs = softmax(s_val_preds)
    return probs[:, :1] / (probs[:, 1:] + 1e-10) * s_N * 1.0 / t_N


def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta, np.mean(weighted_error)


def show_results_dev_iwv_allseeds(lamb_list, base_dir, method='cmd', dataset='painting-quickdraw', seeds=None):
    dev_alg_lmbd = []
    iwv_alg_lmbd = []
    for lamb in lamb_list:
        # RESULTS TRAINED DIRECTLY ON PROXYA FEATURE SPACE
        # load iwv results for domain classifier
        proxya_res = load_results_allseeds(base_dir, f"proxya_pred_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        cls_res = load_results_allseeds(base_dir, f"cls_pred_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        dev_alg = []
        iwv_alg = []
        for id in range(len(proxya_res)):
            s_val_preds = flatten(proxya_res[id]['s_preds'])
            s_N = flatten(proxya_res[id]['s_preds']).shape[0]
            t_N = flatten(proxya_res[id]['t_preds']).shape[0]
            weights = get_weight(s_val_preds, s_N, t_N)
            s_val_preds = flatten(cls_res[id]['s_preds'])
            s_val_lbls = flatten(cls_res[id]['s_lbls'])
            err = np.asarray(1 - (predict(s_val_preds) == s_val_lbls))[:, None]
            dev_risk, importance_weighted = get_dev_risk(weights, err)
            dev_alg.append(dev_risk)
            iwv_alg.append(importance_weighted)

        dev_alg_lmbd.append(dev_alg)
        iwv_alg_lmbd.append(iwv_alg)
    return np.array(dev_alg_lmbd), np.array(iwv_alg_lmbd)


def show_results_bp_proxya_allseeds(lamb_list, base_dir, method='cmd', dataset='painting-quickdraw', seeds=None, B=3, eps=1e-9):
    dist_matrix = {}
    cls_preds = {}
    for lamb in lamb_list:
        # RESULTS TRAINED DIRECTLY ON PROXYA FEATURE SPACE
        # load iwv results for domain classifier
        proxya_dist_res = load_results_allseeds(base_dir, f"proxya_distance_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        cls_pred = load_results_allseeds(base_dir, f"cls_pred_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        # for id in range(len(proxya_dist_res)):
        dist_matrix[lamb] = proxya_dist_res
        cls_preds[lamb] = [flatten(cls_pred[i]['t_preds']) for i in range(len(proxya_dist_res))]

    D_alphas = np.array(list(dist_matrix.values()))
    bp_list = []

    for i, (lamb_i, dist_i) in enumerate(dist_matrix.items()):
        if i == 0: continue
        diff_dist = []
        for j, (lamb_j, dist_j) in enumerate(dist_matrix.items()):
            if j >= i: continue
            emp_norms = np.zeros(len(proxya_dist_res))
            for s in range(len(proxya_dist_res)):
                preds_i = np.argmax(cls_preds[lamb_i][s], axis=1)
                preds_j = np.argmax(cls_preds[lamb_j][s], axis=1)
                emp_norms[s] = 1 - np.mean(np.equal(preds_i, preds_j))

            rhs_lhs = D_alphas[i] * (2 + (2 * B) / (D_alphas[0] + eps)) - emp_norms
            vote_reject = np.zeros(len(rhs_lhs))
            vote_reject[rhs_lhs <= 0] = 1
            diff_dist.append(np.mean(vote_reject))
        bp_list.append(np.max(diff_dist))
    return D_alphas, bp_list


def show_results_bp_mdd_allseeds(lamb_list, base_dir, method='cmd', dataset='painting-quickdraw', seeds=None, gamma=2, B=3, eps=1e-9):
    dist_matrix = {}
    cls_preds = {}
    for lamb in lamb_list:
        # RESULTS TRAINED DIRECTLY ON PROXYA FEATURE SPACE
        # load iwv results for domain classifier
        mdd_dist_res = load_results_allseeds(base_dir, f"mdd_distance_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        cls_pred = load_results_allseeds(base_dir, f"cls_pred_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        dist_matrix[lamb] = [mdd_dist_res[i][f'mdd-dist_gamma-{float(gamma)}']  for i in range(len(mdd_dist_res))]
        cls_preds[lamb] = [flatten(cls_pred[i]['t_preds']) for i in range(len(mdd_dist_res))]

    D_alphas = np.array(list(dist_matrix.values()))
    bp_list = []

    for i, (lamb_i, dist_i) in enumerate(dist_matrix.items()):
        if i == 0: continue
        diff_dist = []
        for j, (lamb_j, dist_j) in enumerate(dist_matrix.items()):
            if j >= i: continue
            emp_norms = np.zeros(len(mdd_dist_res))
            for s in range(len(mdd_dist_res)):
                preds_i = np.argmax(cls_preds[lamb_i][s], axis=1)
                preds_j = np.argmax(cls_preds[lamb_j][s], axis=1)
                emp_norms[s] = 1 - np.mean(np.equal(preds_i, preds_j))

            rhs_lhs = D_alphas[i] * (2 + (2 * B) / (D_alphas[0] + eps)) - emp_norms

            vote_reject = np.zeros(len(rhs_lhs))
            vote_reject[rhs_lhs <= 0] = 1
            diff_dist.append(np.mean(vote_reject))
        bp_list.append(np.max(diff_dist))
    return D_alphas, bp_list


def show_results_cls_risk_allseeds(lamb_list, base_dir, method='cmd', dataset='painting-quickdraw', seeds=None,):

    s_cls_err = {}
    t_cls_err = {}
    for lamb in lamb_list:
        cls_pred = load_results_allseeds(base_dir, f"cls_pred_dataset_{dataset}.npz", method, dataset, f'{dataset}-{lamb}', seeds)
        s_preds = [np.argmax(flatten(cls_pred[i]['s_preds']), axis=1) for i in range(len(cls_pred))]
        t_preds = [np.argmax(flatten(cls_pred[i]['t_preds']), axis=1) for i in range(len(cls_pred))]
        s_lbls = [flatten(cls_pred[i]['s_lbls']) for i in range(len(cls_pred))]
        t_lbls = [flatten(cls_pred[i]['t_lbls']) for i in range(len(cls_pred))]

        s_cls_err[lamb] = np.mean(1 - np.equal(s_preds, s_lbls), axis=1)
        t_cls_err[lamb] = np.mean(1 - np.equal(t_preds, t_lbls), axis=1)

    s_cls_err_arr = np.array(list(s_cls_err.values()))
    t_cls_err_arr = np.array(list(t_cls_err.values()))

    return s_cls_err_arr, t_cls_err_arr


def show_results():
    lamb_list = [
        0,
        0.001,
        0.01,
        0.1,
        1,
        10
    ]

    base_dir = '</path/to/experiments>'
    method = 'mdd'
    domains = ['painting-real-sketch-clipart-infograph-quickdraw',
               'painting-quickdraw-sketch-clipart-infograph-real',
               'painting-quickdraw-real-clipart-infograph-sketch',
               'painting-quickdraw-real-sketch-infograph-clipart',
               'painting-quickdraw-real-sketch-clipart-infograph',
               'quickdraw-real-sketch-clipart-infograph-painting']

    for dataset in domains:
        seeds = ['11223', '213564', '844585']

        dev_alg, iwv_alg = show_results_dev_iwv_allseeds(lamb_list, base_dir, method, dataset, seeds)
        d_proxya, bp_proxya = show_results_bp_proxya_allseeds(lamb_list, base_dir, method, dataset, seeds)
        d_mdd, bp_mdd = show_results_bp_mdd_allseeds(lamb_list, base_dir, method, dataset, seeds, gamma=1.1)
        s_cls_err, t_cls_err = show_results_cls_risk_allseeds(lamb_list, base_dir, method, dataset, seeds)

        extract_risks(lamb_list, seeds, method, dataset,
                      s_cls_err=s_cls_err,
                      t_cls_err=t_cls_err,
                      d_proxya=d_proxya,
                      bp_proxya=bp_proxya,
                      d_mdd=d_mdd,
                      bp_mdd=bp_mdd,
                      iwv_alg=iwv_alg,
                      dev_alg=dev_alg)


if __name__ == "__main__":
    show_results()
