import logging
import os
from abc import ABC, abstractmethod
import csv, os, torch, logging, random

import numpy as np
import lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.trainer import Trainer

from itipy.iti import ITIModule

class BasicPlot(pl.Callback):
    """
        Basic plot callback for visualization of the data and the model predictions.

        Args:
            data (Dataset): Data to visualize.
            model (ITIModule): Model to use.
            plot_id (str): Plot id.
            plot_settings (list): List of plot settings.
            dpi (int): Dots per inch.
            batch_size (int): Batch size.
        """

    def __init__(self, data, model: Trainer, plot_id, plot_settings, dpi=100, batch_size=None, **kwargs):
        self.data = data
        self.model = model
        self.plot_settings = plot_settings
        self.dpi = dpi
        self.plot_id = plot_id
        self.batch_size = batch_size if batch_size is not None else len(data)

        super().__init__(**kwargs)

    def on_validation_epoch_end(self, *args, **kwargs):
        data = self.loadData()

        rows = len(data)
        columns = len(data[0])

        f, axarr = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
        axarr = np.reshape(axarr, (rows, columns))
        for i in range(rows):
            for j in range(columns):
                plot_settings = self.plot_settings[j].copy()
                ax = axarr[i, j]
                ax.axis("off")
                ax.set_title(plot_settings.pop("title", None))
                ax.imshow(data[i][j], **plot_settings)
        plt.tight_layout()
        wandb.log({f"{self.plot_id}": f})
        plt.close()
        del f, axarr, data

    def loadData(self):
        with torch.no_grad():
            loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
            data, predictions = [], []
            for data_batch in loader:
                data_batch = data_batch.float().cuda()
                predictions_batch = self.predict(data_batch)
                data += [data_batch.detach().cpu().numpy()]
                predictions += [[pred.detach().cpu().numpy() for pred in predictions_batch]]
            data = np.concatenate(data)
            predictions = map(list, zip(*predictions)) # transpose
            predictions = [np.concatenate(p) for p in predictions]
            samples = [data, ] + [*predictions]
            # separate into rows and columns
            return [[d[j, i] for d in samples for i in range(d.shape[1])] for j in
                    range(len(self.data))]

    def predict(self, input_data):
        raise NotImplementedError()


class PlotABA(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the translation Instrument A -> Instrument B -> Instrument A.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, save_dir, log_iteration=1000, plot_settings_A=None, plot_settings_B=None, plot_id="ABA",
                 **kwargs):
        self.save_dir = save_dir
        self.log_iteration = log_iteration

        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_A, *plot_settings_B, *plot_settings_A]

        super().__init__(data, model, plot_id, plot_settings, **kwargs)

    def __call__(self, it):
        if it % self.log_iteration:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        imgs = self.loadData()

    def predict(self, x):
        x_ab, x_aba = self.model.forwardABA(x)
        return x_ab, x_aba


class PlotBAB(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the translation Instrument B -> Instrument A -> Instrument B.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, save_dir, log_iteration=1000, plot_settings_A=None, plot_settings_B=None, plot_id="BAB",
                 **kwargs):
        self.save_dir = save_dir
        self.log_iteration = log_iteration

        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) \
            else [plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) \
            else [plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_B, *plot_settings_A, *plot_settings_B]

        super().__init__(data, model, plot_id, plot_settings, **kwargs)

    def __call__(self, it):
        if it % self.log_iteration:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        imgs = self.loadData()

    def predict(self, x):
        x_ba, x_bab = self.model.forwardBAB(x)
        return x_ba, x_bab


class PlotAB(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the translation Instrument A -> Instrument B.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, plot_settings_A=None, plot_settings_B=None, plot_id="AB",
                 **kwargs):
        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_A, *plot_settings_B]

        super().__init__(data, model, path, plot_id, plot_settings, **kwargs)

    def predict(self, input_data):
        x_ab = self.model.forwardAB(input_data)
        return (x_ab,)


class VariationPlotBA(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the variation Instrument B -> Instrument A.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        n_samples (int): Number of samples.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, n_samples, plot_settings_A=None, plot_settings_B=None, plot_id="variation",
                 **kwargs):
        self.n_samples = n_samples

        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_A = plot_settings_A * n_samples
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_B, *plot_settings_A]

        super().__init__(data, model, plot_id, plot_settings, **kwargs)

    def predict(self, x):
        x_ba = torch.cat([self.model.forwardBA(x) for _ in range(self.n_samples)], 1)
        return (x_ba,)

class DummyPlot:
    def __init__(self, *a, **kw): pass
    def __call__(self, it): pass


# class SaveCallback(pl.Callback):
#     """
#     Callback to save the model state and the generator weights.
#
#     Args:
#         checkpoint_dir (str): Directory to save the checkpoints.
#     """
#     def __init__(self, checkpoint_dir):
#         self.checkpoint_dir = checkpoint_dir
#         super().__init__()
#
#     def on_validation_epoch_end(self, trainer: "pl.Trainer", module: "ITIModule") -> None:
#
#         state_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
#         checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{trainer.global_step:06d}.pt')
#         state = {'gen_ab': module.gen_ab,
#                  'gen_ba': module.gen_ba,
#                  'noise_est': module.estimator_noise,
#                  'disc_a': module.dis_a,
#                  'disc_b': module.dis_b,
#                  'state_dict': module.state_dict(),
#                  'global_step': trainer.global_step}
#         torch.save(state, state_path)
#
#         torch.save(state, checkpoint_path)
#         torch.save(module.gen_ab, os.path.join(self.checkpoint_dir, 'generator_AB.pt'))
#         torch.save(module.gen_ba, os.path.join(self.checkpoint_dir, 'generator_BA.pt'))

import os, torch

class SaveCallback:

    def __init__(self, trainer, checkpoint_dir):
        self.trainer = trainer
        self.dir     = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def __call__(self, iteration):
        self.trainer.save(self.dir, iteration)


class HistoryCallback:
    """Save trainer.train_loss to CSV"""
    def __init__(self, trainer, base_dir, log_every=100):
        self.model = trainer
        self.log_every = log_every
        self.csv_path = os.path.join(base_dir, "history", "train_loss.csv")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(trainer.train_loss.keys())

    def __call__(self, iteration):
        if iteration % self.log_every:
            return
        row = [iteration] + [
            getattr(self.model, k, 0.0) for k in self.model.train_loss.keys() if k != "iteration"
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

class ProgressCallback:
    """LOGGER loss to terminal"""
    def __init__(self, trainer, log_every=100):
        self.m = trainer
        self.every = log_every
    def __call__(self, it):
        if it % self.every == 0:
            msg = (f"[{it:6d}] "
                   f"GenA={self.m.loss_gen_a_translate:.3f} "
                   f"GenB={self.m.loss_gen_b_translate:.3f} "
                   f"Dis ={self.m.loss_dis_total:.3f} "
                   f"Div ={self.m.loss_gen_diversity:.3f}")
            logging.info(msg)

import csv, os, torch, logging, random

class ValidationHistoryCallback:

    def __init__(self, trainer, ds_A, ds_B,
                 outdir, log_every=1000, batch_size=1, num_samples=1):
        self.m = trainer
        self.A, self.B = ds_A, ds_B
        self.every = log_every
        self.batch = batch_size
        self.num_samples = num_samples
        self.csv = os.path.join(outdir, "history", "valid_loss.csv")
        os.makedirs(os.path.dirname(self.csv), exist_ok=True)

        with open(self.csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["iter", "L_genA_tr", "L_genB_tr", "L_disA", "L_disB"]
            )

    def __call__(self, it):
        if it % self.every:
            return

        for _ in range(self.num_samples):
            idx_a = random.randrange(len(self.A))
            idx_b = random.randrange(len(self.B))
            x_a = self.A[idx_a].unsqueeze(0).cuda().float()
            x_b = self.B[idx_b].unsqueeze(0).cuda().float()
            self.m.validate(x_a, x_b)

        vals = [it,
                self.m.valid_loss_gen_a_translate.item(),
                self.m.valid_loss_gen_b_translate.item(),
                self.m.valid_loss_dis_a.item(),
                self.m.valid_loss_dis_b.item()]
        with open(self.csv, "a", newline="") as f:
            csv.writer(f).writerow(vals)

        logging.info(
            f"[VAL {it}] GenA_tr={vals[1]:.3f}  GenB_tr={vals[2]:.3f}  "
            f"DisA={vals[3]:.3f}  DisB={vals[4]:.3f}"
        )


from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import datetime, torchvision, numpy as np, torch, os

class TBLossCallback:
    def __init__(self, trainer, logdir, log_every=100):
        self.m = trainer
        os.makedirs(logdir, exist_ok=True)
        self.tb = SummaryWriter(logdir)
        self.every = log_every

    def __call__(self, it):
        if it % self.every:
            return
        self.tb.add_scalar("GenA2B_translate", self.m.loss_gen_a_translate.item(), it)
        self.tb.add_scalar("GenB2A_translate", self.m.loss_gen_b_translate.item(), it)
        self.tb.add_scalar("DiscA",           self.m.loss_dis_a.item(),            it)
        self.tb.add_scalar("DiscB",           self.m.loss_dis_b.item(),            it)

class TBImageCallback:
    def __init__(self, trainer, ds_A, ds_B, period=2000, n_samples=1, max_hw=256,
                 logdir="runs/images"):
        self.m, self.A, self.B = trainer, ds_A, ds_B
        self.period = period
        self.n_samples = n_samples
        self.max_hw = max_hw
        self.tb = SummaryWriter(logdir)

    def __call__(self, it):
        if it % self.period:
            return
        self.m.eval()
        with torch.no_grad():
            xa = torch.from_numpy(self.A[0]).unsqueeze(0).cuda().float()
            xb = torch.from_numpy(self.B[0]).unsqueeze(0).cuda().float()
            xab = self.m.forwardAB(xa)
            xba = self.m.forwardBA(xb)
            # [xa, xab, xb, xba]
            grid = vutils.make_grid(
                torch.cat([xa, xab, xb, xba], 0), nrow=4, normalize=True, value_range=(0,1)
            )
            self.tb.add_image("translation", grid, it)
        self.m.train()

