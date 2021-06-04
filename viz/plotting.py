import torch
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def color_pallet(n_classes, alpha=1.0, darker=0.3):
    cdict = np.array([np.zeros(shape=(4,)) for _ in range(n_classes)])
    parts = [(1. / n_classes)*(i+1) for i in range(n_classes)]
    for i in range(n_classes): 
        idx = (i+1) % (n_classes)
        cdict[i, 0] = np.maximum(parts[idx] - darker, 0.0) # R
        idx = (i+2) % (n_classes)
        cdict[i, 1] = np.maximum(parts[idx] - darker, 0.0) # G
        idx = (i+3) % (n_classes)
        cdict[i, 2] = np.maximum(parts[idx] - darker, 0.0) # B
        cdict[i, 3] = alpha
    return cdict


def plot_moon(x, y, file_name='tmp/plot_moon.png'):
    """Create simple moon plot to showcase data"""
    plt.figure(figsize=(10,10))
    # Draw grid lines with red color and dashed style
    plt.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4)
    n_classes = np.max(y)+1
    cdict = color_pallet(n_classes)
    plt.scatter(x[:, 0], x[:, 1], label=y, c=cdict[y])
    plt.title('Moon Data')
    plt.tight_layout(pad=2.0)
    plt.savefig(file_name)
    plt.close()


def plot_moon_comparison(x1, y1, x2, y2, file_name='tmp/plot_moon_comparison.png'):
    """Create domain shift moon plot to showcase data and domain shifted data"""
    plt.figure(figsize=(10,10))
    # Draw grid lines with red color and dashed style
    plt.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4)
    n_classes = np.max(y1)+1
    cdict1 = color_pallet(n_classes)
    cdict2 = color_pallet(n_classes, alpha=0.5)

    plt.scatter(x1[:,0], x1[:,1], label=y1, c=cdict1[y1])
    plt.scatter(x2[:,0], x2[:,1], label=y2, c=cdict2[y2])
    plt.title('Moon Data')
    plt.tight_layout(pad=2.0)
    plt.savefig(file_name)
    plt.close()


@torch.no_grad()
def plot_moon_decision(device, model, x1, y1, x2, y2, file_name='tmp/plot_moon_decision.png'):
    """Create plot to showcase decision boundry of a NN classifier"""
    plt.figure(figsize=(14,10))

    # set boarder data
    X = x1.astype(np.float32)
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1., X[:, 0].max()+1.
    min2, max2 = X[:, 1].min()-1., X[:, 1].max()+1.
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))
    grid = torch.from_numpy(grid.astype(np.float32)).to(device)
    # make predictions for the grid
    logits = model(grid)
    preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
    yhat = preds.cpu().numpy()

    n_classes = np.max(y1)+1
    cdict1 = color_pallet(n_classes, alpha=0.7)
    cdict2 = color_pallet(n_classes)
    custom_cmap = matplotlib.colors.ListedColormap([cdict1[i] for i in range(cdict1.shape[0])])
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap=custom_cmap, alpha=0.25)

    plt.scatter(x1[:,0], x1[:,1], label=y1, c=cdict1[y1], marker='o')
    plt.scatter(x2[:,0], x2[:,1], label=y2, c=cdict2[y2], marker='*')
    plt.title('Moon Data')
    source_domain = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                  markersize=10, label='source domain')
    target_domain = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                                  markersize=10, label='target domain')

    plt.legend(handles=[source_domain, target_domain])
    plt.tight_layout(pad=2.0)
    plt.savefig(file_name)
    plt.close()


def twinplot_metrics(train_losses, eval_losses, train_accs, eval_accs, titel=None, plot_name1='Loss', plot_name2='Accuracy', file_name='tmp/twinplot_metrics.png'):
    """Create two plots side-by-side of the loss and accuracy curves"""
    fig = plt.figure(figsize=(16,6))
    fig.suptitle(titel, fontsize=16)
    ax1 = fig.add_subplot(121, label=plot_name1)
    ax2 = fig.add_subplot(122, label=plot_name2)

    # Draw grid lines with red color and dashed style
    ax1.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4, axis='y')
    ax1.plot(train_losses, color="C0", label='train')
    ax1.plot(eval_losses, color="C1", label='eval')
    ax1.set_ylabel(plot_name1, color="black")
    ax1.set_xlabel("Epoch", color="black")
    ax1.tick_params(axis='x', colors="black")
    ax1.tick_params(axis='y', colors="black")
    ax1.legend()

    # Draw grid lines with red color and dashed style
    ax2.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4, axis='y')
    ax2.plot(train_accs, color="C0", label='train')
    ax2.plot(eval_accs, color="C1", label='eval')
    ax2.set_ylabel(plot_name2, color="black")
    ax2.set_xlabel("Epoch", color="black")
    ax2.tick_params(axis='x', colors="black")
    ax2.tick_params(axis='y', colors="black")
    ax2.legend()

    fig.tight_layout(pad=2.0)
    fig.savefig(file_name)
    plt.close(fig)


def ci_twinplot_metrics(eval_losses, eval_accs, titel=None, plot_name1='Loss', plot_name2='Accuracy', file_name='tmp/ci_twinplot_metrics.png'):
    """Create plot to show confidence interval of the evaluation loss and accuarcy"""
    fig = plt.figure(figsize=(16,6))
    fig.suptitle(titel, fontsize=16)
    ax1 = fig.add_subplot(121, label=plot_name1)
    ax2 = fig.add_subplot(122, label=plot_name2)

    # Draw grid lines with red color and dashed style
    ax1.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4, axis='y')
    eval_losses_mean, eval_losses_ub, eval_losses_lb = eval_losses
    ax1.fill_between(range(eval_losses_mean.shape[0]), eval_losses_ub, eval_losses_lb,
                     color="C1", alpha=.3)
    ax1.plot(eval_losses_mean, color="C1", label='eval', marker='o')
    ax1.set_ylabel(plot_name1, color="black")
    ax1.set_xlabel("Epoch", color="black")
    ax1.tick_params(axis='x', colors="black")
    ax1.tick_params(axis='y', colors="black")
    ax1.legend()

    # Draw grid lines with red color and dashed style
    ax2.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4, axis='y')
    eval_accs_mean, eval_accs_ub, eval_accs_lb = eval_accs
    ax2.fill_between(range(eval_accs_mean.shape[0]), eval_accs_ub, eval_accs_lb,
                     color="C1", alpha=.3)
    ax2.plot(eval_accs_mean, color="C1", label='eval', marker='o')
    ax2.set_ylabel(plot_name2, color="black")
    ax2.set_xlabel("Epoch", color="black")
    ax2.tick_params(axis='x', colors="black")
    ax2.tick_params(axis='y', colors="black")
    ax2.legend()

    fig.tight_layout(pad=2.0)
    fig.savefig(file_name)
    plt.close(fig)


def singleplot_values(values_list, titel=None, plot_name='Score', file_name='tmp/singleplot_values.png'):
    """Create a single plot for a generic value"""
    fig = plt.figure(figsize=(16,6))
    fig.suptitle(titel, fontsize=16)
    ax1 = fig.add_subplot(111, label=plot_name)

    # Draw grid lines with red color and dashed style
    ax1.grid(color='grey', linestyle='-', linewidth=0.7, alpha=0.4, axis='y')
    for i, (key, values) in enumerate(values_list.items()):
        ax1.plot(values, color=f"C{i}", label=key)

    ax1.set_ylabel(plot_name, color="black")
    ax1.set_xlabel("Epoch", color="black")
    ax1.tick_params(axis='x', colors="black")
    ax1.tick_params(axis='y', colors="black")
    ax1.legend()

    fig.tight_layout(pad=2.0)
    fig.savefig(file_name)
    plt.close(fig)
