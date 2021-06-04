import torch
import copy
import os
import time
import uuid
import numpy as np
from tqdm import tqdm
from lighter.collectible import BaseCollectible
from lighter.writer import BaseWriter
from lighter.misc import generate_long_id
from viz.plotting import twinplot_metrics, ci_twinplot_metrics
from misc.helpers import load_function, map_reduce, set_seed, load_seed_list, EarlyStopping
from torch.cuda.amp import autocast, GradScaler


def compute_alpha(config, loader, i, epoch, gamma=10):
    p = float(i + epoch * len(loader)) / config.trainer.epochs / len(loader)
    alpha = 2. / (1. + np.exp(-gamma * p)) - 1
    return alpha


def lr_lambda(mu0=0.01, alpha=10, beta=0.75):
    def compute_lr(epoch):
        return mu0/((1+alpha*epoch)**beta)
    return compute_lr


class Trainer(object):
    r"""Main training loop for the current approach.
    """
    def __init__(self, config, model, dataloaders, info='', start_epoch=0, verbose=False):
        self.config = config
        self.model = model
        self.verbose = verbose
        if len(dataloaders) == 2:
            self.train_loader, self.eval_loader = dataloaders
        elif len(dataloaders) == 3:
            self.train_loader, self.eval_loader, self.test_loader = dataloaders
        self.save_interval = config.trainer.save_interval
        # handle experiment related variables
        self.experiment_id = f"{config.trainer.experiment_name}-{info}/seed{config.seed}-{generate_long_id()}"
        self.checkpoint_dir = os.path.join(config.trainer.checkpoint_dir, 
                                           self.experiment_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = start_epoch
        print(f'Started experiment: {self.experiment_id}')
        # used for collecting data statistics
        self.collectible = BaseCollectible()
        # used to write to tensorboard
        self.tensorboard_dir = os.path.join(config.trainer.tensorboard_dir, self.experiment_id)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = BaseWriter(log_dir=self.tensorboard_dir)
        # load training related metric, optimizer and criterion functions
        config.approach.lr_lambda = lr_lambda
        self.optimizer, self.scheduler = load_function(config.trainer.optimizer_file, config.trainer.optimizer)(config, self.model)
        config.approach.lr_lambda = None
        self.metric = load_function(config.trainer.metric_file, config.trainer.metric)
        self.criterion = load_function(config.trainer.criterion_file, config.trainer.criterion)
        # initialize early stopping
        self.early_stopping = EarlyStopping(self, patience=config.trainer.early_stopping_patience, 
                                            min_epochs=config.trainer.min_epochs, 
                                            verbose=verbose)
        # Creates a GradScaler once at the beginning of training.
        self.scaler = GradScaler()

    def save_checkpoint(self, epoch, info=''):
        """Saves a model checkpoint"""
        conf = copy.deepcopy(self.config)
        conf.device = None # required because device is not serializable
        state = {
            'info': info,
            'epoch': epoch,
            'experiment_id': self.experiment_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': conf
        }
        try:
            os.mkdir(self.checkpoint_dir)
        except FileExistsError:
            pass
        filename = os.path.join(self.checkpoint_dir, f'{info}checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        if self.verbose: print("Saving checkpoint: {} ...".format(filename))

    def resume_checkpoint(self, resume_path):
        """Resumes training from an existing model checkpoint"""
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    @torch.enable_grad()
    def _train(self, epoch):
        """Training step through the batch"""
        losses = []
        accs = []
        values = []
        self.model.train()
        for i, (xs, ys, xt, yt) in enumerate(self.train_loader):
            self.config.source_only = False
            # create alpha value for DANN according to paper https://arxiv.org/abs/1505.07818
            alpha = compute_alpha(self.config, self.train_loader, i, epoch)
            # prepare input
            if self.config.trainer.criterion == "source_only":
                self.config.source_only = True
                input = xs.to(self.config.device)
            elif self.config.approach.lambda_ == 0:
                input = torch.cat([xs.to(self.config.device), xs.to(self.config.device)], dim=0)
            else:
                input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            s_target = ys.to(self.config.device)
            t_target = yt.to(self.config.device)
            # reset gradients
            self.optimizer.zero_grad()
            # Runs the forward pass with autocasting for half precision.
            with autocast():
                # forward through the model
                output = self.model(input, alpha)
                # compute loss
                loss, loss_items = self.criterion(self.config, output, s_target, t_target)
            if self.config.trainer.use_mixed_precission:
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if self.config.trainer.apply_gradient_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_gradient_norm)
            # evaluate metrics
            acc = self.metric(self.config, output, s_target, t_target)
            # collect the stats
            accs.append(acc)
            losses.append(loss_items)
            vals = {'alpha': alpha}
            values.append(vals)
            if self.config.trainer.use_mixed_precission:
                # perform optimization step
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                self.scaler.update()
            else:
                self.optimizer.step()
            # update stats for tensorboard
            self.collectible.update(category='train', **loss_items)
            self.collectible.update(category='train', **acc)
            self.collectible.update(category='train', **vals)
        return {'losses': losses, 'accs': accs, 'values': values}

    @torch.no_grad()
    def _eval(self, epoch):
        """Evaluation step through the batch"""
        losses = []
        accs = []
        values = []
        self.model.eval()
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            self.config.source_only = False
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            s_target = ys.to(self.config.device)
            t_target = yt.to(self.config.device)
            # forward through the model
            output = self.model(input)
            # compute loss
            loss, loss_items = self.criterion(self.config, output, s_target, t_target)
            # evaluate metrics
            acc = self.metric(self.config, output, s_target, t_target)
            # collect the stats
            accs.append(acc)
            losses.append(loss_items)
            vals = {'alpha': 1.0}
            values.append(vals)
            # update stats for tensorboard
            self.collectible.update(category='eval', **loss_items)
            self.collectible.update(category='eval', **acc)
        return {'loss': loss, 'losses': losses, 'accs': accs, 'values': values}

    @torch.no_grad()
    def mdd(self):
        """Compute mdd distances"""
        self.model.eval()
        distances = {}
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            s_target = ys.to(self.config.device)
            t_target = yt.to(self.config.device)
            # forward through the model
            output = self.model(input)
            # evaluate metrics
            metric = self.metric(self.config, output, s_target, t_target)
            # collect the stats
            for key in [key for key in metric.keys() if 'mdd-dist' in key.lower()]:
                distances[key] = metric[key]
        return distances

    @torch.no_grad()
    def proxya(self):
        """Compute proxy-a distances"""
        self.model.eval()
        accs = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            s_target = ys.to(self.config.device)
            t_target = yt.to(self.config.device)
            # forward through the model
            output = self.model(input)
            # evaluate metrics
            acc = self.metric(self.config, output, s_target, t_target)
            # collect the stats
            accs.append(acc['acc'])
        err = 2 * (1 - 2 * (1 - np.mean(accs)))
        return err

    @torch.no_grad()
    def val_cls_preds(self, include_da_preds=True):
        self.model.eval()
        s_preds = []
        t_preds = []
        s_da_preds = []
        t_da_preds = []
        s_lbls = []
        t_lbls = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            b = xs.shape[0]
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            # forward through the model
            classifier_preds, da_preds = self.model(input)
            # predictions
            s_pred = classifier_preds[:b].cpu().numpy()
            t_pred = classifier_preds[b:].cpu().numpy()
            s_preds.append(s_pred)
            t_preds.append(t_pred)
            s_lbls.append(ys.cpu().numpy())
            t_lbls.append(yt.cpu().numpy())
            if include_da_preds:
                da_source_pred = da_preds[:b].cpu().numpy()
                da_target_pred = da_preds[b:].cpu().numpy()            
                s_da_preds.append(da_source_pred)
                t_da_preds.append(da_target_pred)
        res = {
            's_preds': s_preds, 
            't_preds': t_preds, 
            's_lbls': s_lbls,
            't_lbls': t_lbls
        }
        if include_da_preds:
            res['s_da_preds'] = s_da_preds
            res['t_da_preds'] = t_da_preds
        return res

    @torch.no_grad()
    def val_proxya_preds(self):
        self.model.eval()
        s_preds = []
        t_preds = []
        s_lbls = []
        t_lbls = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            b = xs.shape[0]
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            # forward through the model
            classifier_preds, da_preds = self.model(input)
            # predictions
            s_pred = classifier_preds[:b].cpu().numpy()
            t_pred = classifier_preds[b:].cpu().numpy()
            s_preds.append(s_pred)
            t_preds.append(t_pred)
            s_lbls.append(torch.zeros_like(ys).numpy())
            t_lbls.append(torch.ones_like(yt).cpu().numpy())
        res = {
            's_preds': s_preds, 
            't_preds': t_preds, 
            's_lbls': s_lbls,
            't_lbls': t_lbls
        }
        return res

    @torch.no_grad()
    def val_iwv_preds(self):
        self.model.eval()
        s_preds = []
        t_preds = []
        s_lbls = []
        t_lbls = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            b = xs.shape[0]
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            # forward through the model
            classifier_preds, da_preds = self.model(input)
            # predictions
            s_pred = classifier_preds[:b].cpu().numpy()
            t_pred = classifier_preds[b:].cpu().numpy()
            s_preds.append(s_pred)
            t_preds.append(t_pred)
            s_lbls.append(torch.zeros_like(ys).numpy())
            t_lbls.append(torch.ones_like(yt).cpu().numpy())
        res = {
            's_preds': s_preds, 
            't_preds': t_preds, 
            's_lbls': s_lbls,
            't_lbls': t_lbls
        }
        return res

    @torch.no_grad()
    def val_mdd_preds(self):
        self.model.eval()
        s_preds = []
        t_preds = []
        s_lbls = []
        t_lbls = []
        for i, (xs, ys, xt, yt) in enumerate(self.eval_loader):
            b = xs.shape[0]
            # prepare input
            input = torch.cat([xs.to(self.config.device), xt.to(self.config.device)], dim=0)
            # forward through the model
            _, mdd_preds = self.model(input)
            # predictions
            s_pred = mdd_preds[:b].cpu().numpy()
            t_pred = mdd_preds[b:].cpu().numpy()
            s_preds.append(s_pred)
            t_preds.append(t_pred)
            s_lbls.append(ys.cpu().numpy())
            t_lbls.append(yt.cpu().numpy())
        res = {
            's_preds': s_preds, 
            't_preds': t_preds, 
            's_lbls': s_lbls,
            't_lbls': t_lbls
        }
        return res

    def run(self):
        """Main run loop over multiple epochs"""
        train_losses = []
        eval_losses = []
        train_accs = []
        eval_accs = []
        train_values = []
        eval_values = []
        self.start_epoch += 1
        # initialize progress bar
        with tqdm(range(self.start_epoch, self.config.trainer.epochs + 1)) as pbar:
            # run over n epochs
            for epoch in pbar:
                # check eval time
                start_time = time.time_ns()
                # perform an evaluation step
                eval_res = self._eval(epoch)
                eval_losses.append(eval_res['losses'])
                eval_accs.append(eval_res['accs'])
                eval_values.append(eval_res['values'])
                eval_time = (time.time_ns()-start_time)/(10**9)

                # check train time
                start_time = time.time_ns()
                # perform an training step
                train_res = self._train(epoch)
                train_losses.append(train_res['losses'])
                train_accs.append(train_res['accs'])
                train_values.append(train_res['values'])
                train_time = (time.time_ns()-start_time)/(10**9)

                # update the progress bar info
                pbar.set_postfix(train_loss=map_reduce(train_losses[-1], 'loss'), 
                                 eval_loss=map_reduce(eval_losses[-1], 'loss'),
                                 train_acc=map_reduce(train_accs[-1], 'acc'), 
                                 eval_acc=map_reduce(eval_accs[-1], 'acc'),
                                 eval_time=eval_time,
                                 train_time=train_time,
                                 refresh=False)

                # if a scheduler is used, perform scheduler step
                if self.scheduler:
                    # check if we are still improving on the source loss otherwise decrease LR
                    self.scheduler.step()

                # create checkpoints perodically
                if self.save_interval != 0 and epoch % self.save_interval == 0:
                    self.save_checkpoint(epoch)

                # summaries the collected stats
                collection = self.collectible.redux()
                # write to tensorboard
                self.writer.write(category='train', **collection)
                self.writer.write(category='eval', **collection)

                # update progress bar step
                pbar.update()
                # update tensorboard counter
                self.writer.step()
                # reset collected stats
                self.collectible.reset()

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                self.early_stopping(epoch, map_reduce(eval_losses[-1], 'loss'))
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

        print(f"Train loss: {map_reduce(train_losses[-1], 'loss')} Eval loss: {map_reduce(eval_losses[-1], 'loss')}")
        print(f"Train accuracy: {map_reduce(train_accs[-1], 'acc')} Eval accuracy: {map_reduce(eval_accs[-1], 'acc')}")
        return {'train_losses': train_losses, 'eval_losses': eval_losses, 'train_accs': train_accs, 'eval_accs': eval_accs, 'train_values': train_values, 'eval_values': eval_values}


def experiments(config):
    """Main experiment entrance point for each `<approach>.py`"""
    # check and set device
    if "device" in config and config.device is not None:
        device = torch.device(config.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Using device: {device}")
    # load seeding list
    if "seed_list" in config and config.seed_list is not None:
        seeds = config.seed_list
        print("Using config seeds", seeds)
    else:
        seeds = load_seed_list()
        print("Using file loaded seeds", seeds)
    lambda_list = config.approach.lambda_list

    # create the dataset and get the dataloader
    config.backbone = config.bp_backbone
    create_domain_adaptation_data = load_function(config.dataloader.module, config.dataloader.funcname)
    for train_loader, eval_loader in create_domain_adaptation_data(config):
        # dataset options
        dataloaders = (train_loader, eval_loader)
        ds_name = f"{train_loader.dataset.source_domain_name}-{train_loader.dataset.target_domain_name}"

        cls_results = {}
        mdd_results = {}
        iwv_results = {}
        proxya_results = {}

        proxya_dist_matrix = {}
        mdd_dist_matrix = {}

        cls_predictions = {}
        proxya_predictions = {}
        mdd_predictions = {}
        iwv_predictions = {}

        # run an experiment for multiple seeds
        for i, seed in enumerate(seeds):
            print(f'Running experiment with seed: {seed} Run: {i+1}/{len(seeds)}')
            set_seed(seed)
            config.seed = seed
            experiment_name = config.trainer.experiment_name
            # create method experiment config
            cfg = config.method
            cfg.debug = config.debug
            cfg.device = device
            cfg.seed = seed
            seed = str(seed)
            cfg.checkpoint = config.checkpoint
            cfg.trainer.epochs = config.trainer.epochs_cls
            cfg.trainer.save_interval = config.trainer.save_interval

            # measure elapsed time per experiment
            start = time.time()

            # run over multiple lambdas
            for k, lamb in enumerate(lambda_list):
                key = f"{ds_name}-{lamb}"
                id = uuid.uuid1()
                print(f'Current domain task {k+1}/{len(lambda_list)}: src to trg', key, 'Experiment id-pair', id)

                # load a model architecture
                Net = load_function(cfg.model.module, cfg.model.classname)
                net = Net(cfg).to(device)
                cfg.trainer.experiment_name = f"{experiment_name}"
                cfg.approach.lambda_ = lamb
                # create a trainer instance and execute the approach
                trainer = Trainer(cfg, net, dataloaders, info=f'method_{id}-{key}')
                if cfg.checkpoint:
                    trainer.resume_checkpoint(cfg.checkpoint)
                cls_res = trainer.run()
                cls_preds = trainer.val_cls_preds(include_da_preds=True)

                # balancing principle classifier Proxy-A
                config.model = config.proxya_model
                config.backbone = config.bp_backbone
                BPNet = load_function(config.model.module, config.model.classname)
                dp_net = BPNet(config).to(device)
                dp_net.set_backbone(net)
                # set main criterion and metric
                config.trainer.criterion = config.trainer.proxya_criterion
                config.trainer.metric = config.trainer.proxya_metric
                # create trainer object
                proxya_trainer = Trainer(config, dp_net, dataloaders, info=f'proxya_{id}-{key}')
                proxya_res = proxya_trainer.run()
                # distance matrix computation and logits
                proxya_dist = proxya_trainer.proxya()
                proxya_preds = proxya_trainer.val_proxya_preds()

                # balancing principle classifier MDD
                config.model = config.mdd_model
                config.backbone = config.bp_backbone
                BPNet = load_function(config.model.module, config.model.classname)
                dp_net = BPNet(config).to(device)
                dp_net.set_backbone(net)
                # set main criterion and metric
                config.trainer.criterion = config.trainer.mdd_criterion
                config.trainer.metric = config.trainer.mdd_metric
                # create trainer object
                mdd_trainer = Trainer(config, dp_net, dataloaders, info=f'mdd-dist_{id}-{key}')
                mdd_res = mdd_trainer.run()
                # distance matrix computation and logits
                mdd_dist = mdd_trainer.mdd()
                mdd_preds = mdd_trainer.val_mdd_preds()

                # init dicts
                if key not in cls_results:
                    cls_results[key] = {}
                if seed not in cls_results[key]:
                    cls_results[key][seed] = []
                if key not in mdd_results:
                    mdd_results[key] = {}
                if seed not in mdd_results[key]:
                    mdd_results[key][seed] = []
                if key not in proxya_results:
                    proxya_results[key] = {}
                if seed not in proxya_results[key]:
                    proxya_results[key][seed] = []

                if key not in proxya_dist_matrix:
                    proxya_dist_matrix[key] = {}
                if seed not in proxya_dist_matrix[key]:
                    proxya_dist_matrix[key][seed] = []
                if key not in mdd_dist_matrix:
                    mdd_dist_matrix[key] = {}
                if seed not in mdd_dist_matrix[key]:
                    mdd_dist_matrix[key][seed] = []
                
                if key not in cls_predictions:
                    cls_predictions[key] = {}
                if seed not in cls_predictions[key]:
                    cls_predictions[key][seed] = []
                if key not in proxya_predictions:
                    proxya_predictions[key] = {}
                if seed not in proxya_predictions[key]:
                    proxya_predictions[key][seed] = []
                if key not in mdd_predictions:
                    mdd_predictions[key] = {}
                if seed not in mdd_predictions[key]:
                    mdd_predictions[key][seed] = []

                cls_results[key][seed].append(cls_res)
                cls_predictions[key][seed].append(cls_preds)
                proxya_results[key][seed].append(proxya_res)
                mdd_results[key][seed].append(mdd_res)
                proxya_dist_matrix[key][seed].append(proxya_dist)
                mdd_dist_matrix[key][seed].append(mdd_dist)
                proxya_predictions[key][seed].append(proxya_preds)
                mdd_predictions[key][seed].append(mdd_preds)

            # importance weighted validation classifier 
            id = uuid.uuid1()
            config.model = config.iwv_model
            config.backbone = config.iwv_backbone
            IWVNet = load_function(config.model.module, config.model.classname)
            iwv_net = IWVNet(config).to(device)
            # set main criterion and metric
            config.trainer.criterion = config.trainer.iwv_criterion
            config.trainer.metric = config.trainer.iwv_metric
            # create trainer object
            iwv_trainer = Trainer(config, iwv_net, dataloaders, info=f'iwv-dist_{id}-{ds_name}')
            iwv_res = iwv_trainer.run()
            # predict iwv validation
            iwv_preds = iwv_trainer.val_iwv_preds()

            if ds_name not in iwv_predictions:
                iwv_predictions[ds_name] = {}
            if seed not in iwv_predictions[ds_name]:
                iwv_predictions[ds_name][seed] = []
            if ds_name not in iwv_results:
                iwv_results[ds_name] = {}
            if seed not in iwv_results[ds_name]:
                iwv_results[ds_name][seed] = []
            
            iwv_results[ds_name][seed].append(iwv_res)
            iwv_predictions[ds_name][seed].append(iwv_preds)

            # check elapsed time
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Task completed - time elapsed for task {} with seed {}> {:0>2}:{:0>2}:{:05.2f}".format(ds_name, seed, int(hours),int(minutes),seconds))

        # save proxya distance matrix
        distance_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'proxya_distance_dataset_{ds_name}.npz')
        np.savez(distance_file, proxya_dist_matrix)
        # save mdd distance matrix
        distance_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'mdd_distance_dataset_{ds_name}.npz')
        np.savez(distance_file, mdd_dist_matrix)

        # save the predictions for cls
        pred_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'cls_pred_dataset_{ds_name}.npz')
        np.savez(pred_file, cls_predictions)
        # save the predictions for proxya
        pred_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'proxya_pred_dataset_{ds_name}.npz')
        np.savez(pred_file, proxya_predictions)
        # save the predictions for mdd
        pred_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'mdd_pred_dataset_{ds_name}.npz')
        np.savez(pred_file, mdd_predictions)
        # save the predictions for iwv
        pred_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'iwv_pred_dataset_{ds_name}.npz')
        np.savez(pred_file, iwv_predictions)

        # save final / total cls_results
        cls_res_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'cls_results_dataset_{ds_name}.npz')
        np.savez(cls_res_file, cls_results)
        # save final / total proxya_results
        mdd_res_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'proxya_results_dataset_{ds_name}.npz')
        np.savez(mdd_res_file, proxya_results)
        # save final / total mdd_results
        mdd_res_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'mdd_results_dataset_{ds_name}.npz')
        np.savez(mdd_res_file, mdd_results)
        # save final / total iwv_results
        mdd_res_file = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name, f'iwv_results_dataset_{ds_name}.npz')
        np.savez(mdd_res_file, iwv_results)

        # reset memory to avoid compounding dataset allocations
        if config.dataloader.DomainNet.reset_and_reload_memory:
            train_loader.dataset.reset_memory()
            eval_loader.dataset.reset_memory()
