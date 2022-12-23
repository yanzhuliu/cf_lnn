"""
code modified from https://github.com/Physics-aware-AI/DiffCoSim/
"""

from argparse import ArgumentParser, Namespace
import os, sys
import json
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from operator import add
from functools import reduce
import os
print(os.getpid())
print(os.getcwd())


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import _logger as log
from pytorch_lightning.overrides.data_parallel import *

# local application imports
from datasets.datasets import RigidBodyDataset
from systems.bouncing_point_masses import BouncingPointMasses
from systems.chain_pendulum_with_contact import ChainPendulumWithContact
from ode_models.ode_w_contact_models import CFLNN
from utils import str2bool
import models

seed_everything(0)

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def collect_tensors(field, outputs):
    res = torch.stack([log[field] for log in outputs], dim=0)
    if res.ndim == 1:
        return res
    else:
        return res.flatten(0, 1)


class Model(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        hparams = Namespace(**hparams) if type(hparams) is dict else hparams
        vars(hparams).update(**kwargs)

        if hparams.body_kwargs_file == "":
            body = str_to_class(hparams.body_class)()
        else:
            with open(os.path.join(THIS_DIR, "examples", hparams.body_kwargs_file+".json"), "r") as file:
                body_kwargs = json.load(file)
            body = str_to_class(hparams.body_class)(hparams.body_kwargs_file, 
                                                    is_reg_data=hparams.is_reg_data,
                                                    is_reg_model=hparams.is_reg_model, 
                                                    is_lcp_data=hparams.is_lcp_data,
                                                    is_lcp_model=hparams.is_lcp_model,
                                                    **body_kwargs)

            # only for counterfactual data generation
            with open(os.path.join(THIS_DIR, "examples", hparams.body_kwargs_file+"_prime.json"), "r") as file:
                cf_body_kwargs = json.load(file)
            cf_body = str_to_class(hparams.body_class)(hparams.body_kwargs_file+"_prime",
                                                    is_reg_data=hparams.is_reg_data,
                                                    is_reg_model=hparams.is_reg_model,
                                                    is_lcp_data=hparams.is_lcp_data,
                                                    is_lcp_model=hparams.is_lcp_model,
                                                    **cf_body_kwargs)

            vars(hparams).update(**body_kwargs)
        vars(hparams).update(
            dt=body.dt, 
            integration_time=body.integration_time,
            is_homo=body.is_homo,
            body=body
        )

        # load/generate data
        train_dataset = str_to_class(hparams.dataset_class)(
            mode = "train",
            n_traj = hparams.n_train,
            body = body,
            cf_body = cf_body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
            noise_std = hparams.noise_std,
        )

        val_dataset = str_to_class(hparams.dataset_class)(
            mode = "val",
            n_traj = hparams.n_val,
            body = body,
            cf_body=cf_body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
            noise_std = hparams.noise_std,
        )

        test_dataset = str_to_class(hparams.dataset_class)(
            mode = "test",
            n_traj = hparams.n_test,
            body = body,
            cf_body=cf_body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
            noise_std = hparams.noise_std,
        )

        datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        if hparams.body_class == "ChainPendulumWithContact" and body.n_c > cf_body.n_c:
            body = cf_body

        ode_func = str_to_class(hparams.network_class)(body_graph=body.body_graph,
                                                         impulse_solver=body.impulse_solver,
                                                         d=body.d,
                                                         n_c=body.n_c,
                                                         device=self.device,
                                                         dtype=torch.float32,
                                                         **vars(hparams))
        self.model = models.VariationalODE(input_dim=1, output_dim=1, ode_func=ode_func, **vars(hparams))
        self.save_hyperparameters(hparams)
        self.body = body
        self.datasets = datasets

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_class)(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay
        )
        if self.hparams.SGDR:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], num_workers=64, pin_memory=True,
                    batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], num_workers=64, pin_memory=True,
                    batch_size=self.hparams.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.datasets["test"], num_workers=64, pin_memory=True,
                    batch_size=self.hparams.batch_size, shuffle=False)

    def traj_mae(self, pred_zts, true_zts):
        return (pred_zts - true_zts).abs().mean()

    def one_batch(self, batch, batch_idx):
        # z0: (bs, 2, n, d), zts: (bs, T, 2, n, d), ts: (bs, T) from trajectory x
        # z0_: (bs, 2, n, d), zts_: (bs, T, 2, n, d), ts: (bs, T) from trajectory x'
        (z0, ts), zts, is_clds, (z0_, ts_), zts_, is_clds_ = batch
        cf3 = False
        if zts_.shape[3] < zts.shape[3]:  # type III cf
            if hparams.body_class == "ChainPendulumWithContact":
                zts = zts[:,:,:,:-1,:]
                z0 = zts[:,0]
            #    is_clds = is_clds[:,:-1]
                cf3 = False
            else:  # for bp5 now
                zts_ = torch.cat([zts_, torch.zeros(zts_.shape).to(zts_).sum(3,keepdim=True)], dim=3)  # add zeros (bs, T, 2, 1, d)
                z0_ = zts_[:,0]
                cf3 = True

        treat = False
        if "cf_2" in self.hparams.body_kwargs_file:
            treat = True

        ts = ts[0] - ts[0,0]
        ts_ = ts_[0] - ts_[0,0]

        hat_zts, hat_zts_, hat_d_zts_ = self.model(z0, zts, is_clds, ts, z0_, ts_, tol=self.hparams.tol, method=self.hparams.solver, cf3=cf3, treat=treat)
        loss, mse = self.model.compute_losses(zts, hat_zts, zts_, hat_zts_, hat_d_zts_)
        self.log("train_loss",loss, on_step = True, on_epoch = True)
        self.log("train_rmse",mse, on_step = True, on_epoch = True)

        return loss, mse

    def training_step(self, batch, batch_idx):
        loss, mse = self.one_batch(batch, batch_idx)
        self.log("train", loss, prog_bar=True)
        self.log("train/mse", mse, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, mse = self.one_batch(batch, batch_idx)
        return mse

    def validation_epoch_end(self, outputs):
        val_loss = reduce(add, outputs) / len(outputs)
        self.log("val/mse", val_loss, prog_bar=True)

        if self.body.is_homo and self.hparams.network_class in ['CLNNwC', 'CHNNwC']:
            mu = F.relu(self.model.ode.mu_params)
            cor = F.hardsigmoid(self.model.ode.cor_params)
            self.log("mu", mu, prog_bar=True)
            self.log("cor", cor, prog_bar=True)

    def test_step(self, batch, batch_idx, integration_time=None):
        (z0, ts), zts, is_clds, (z0_, ts_), zts_, is_clds_ = batch
        if integration_time is None:
            integration_time = max(self.body.integration_time, self.body.dt*100)
        ts = torch.arange(0.0, integration_time, self.body.dt).type_as(z0)
        true_zts = zts_[:,:len(ts)]
        cf3 = False
        if zts_.shape[3] < zts.shape[3]:  # type III cf
            if self.hparams.body_class == "ChainPendulumWithContact":
                zts = zts[:, :, :, :-1, :]
                cf3 = False
            else:  # for bp5 now
                zts_ = torch.cat([zts_, torch.zeros(zts_.shape).to(zts_).sum(3, keepdim=True)],
                                 dim=3)  # add zeros (bs, T, 2, 1, d)
                z0_ = zts_[:, 0]
                cf3 = True

        pred_zts = self.model.ode.integrate(z0_, ts, method='rk4',cf3=cf3)
        if pred_zts.shape[3] > true_zts.shape[3]:
            pred_zts = pred_zts[:,:,:,:-1,:]

        sq_diff = (pred_zts - true_zts).pow(2).sum((2,3,4))
        sq_true = true_zts.pow(2).sum((2,3,4))
        sq_pred = pred_zts.pow(2).sum((2,3,4))
        # (bs, T)
        rel_err = sq_diff.div(sq_true).sqrt()
        bounded_rel_err = sq_diff.div(sq_true+sq_pred).sqrt()
        abs_err = sq_diff.sqrt()
        diff_ = (pred_zts - true_zts).abs().sum((2,3,4))
        true_ = true_zts.abs().sum((2,3,4))
        mape = diff_.div(true_)

        loss = self.traj_mae(pred_zts, true_zts)

        return {
            "traj_mae": loss.detach(),
            "true_zts": true_zts.detach(),
            "pred_zts": pred_zts.detach(),
            "abs_err": abs_err.detach(),
            "rel_err": rel_err.detach(),
            "mape_err": mape.detach(),
            "bounded_rel_err": bounded_rel_err.detach(),
        }

    def test_epoch_end(self, outputs):
        log, save = self._collect_test_steps(outputs)
        self.log("test_loss", log["traj_mae"])
        for k, v in log.items():
            self.log(f"test/{k}", v)

    def _collect_test_steps(sef, outputs):
        loss = collect_tensors("traj_mae", outputs).mean(0).item()
        # collect batch errors from minibatches (BS, T)
        abs_err = collect_tensors("abs_err", outputs)
        rel_err = collect_tensors("rel_err", outputs)
        bounded_rel_err = collect_tensors("bounded_rel_err", outputs)

        pred_zts_true_energy = collect_tensors("pred_zts_true_energy", outputs) # (BS, T)
        true_zts_true_energy = collect_tensors("true_zts_true_energy", outputs)

        true_zts = collect_tensors("true_zts", outputs)
        pred_zts = collect_tensors("pred_zts", outputs)

        log = {
            "traj_mae" : loss,
            "mean_abs_err": abs_err.sum(1).mean(0),
            "mean_rel_err": rel_err.sum(1).mean(0),
            "mean_bounded_rel_err": bounded_rel_err.sum(1).mean(0),
            "mean_true_zts_true_energy": true_zts_true_energy.sum(1).mean(0),
            "mean_pred_zts_true_energy": pred_zts_true_energy.sum(1).mean(0),
        }
        save = {"true_zts": true_zts, "pred_zts": pred_zts}
        return log, save

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # dataset 
        parser.add_argument("--body-class", type=str, choices=["BouncingPointMasses", "ChainPendulumWithContact"],
                            default="BouncingPointMasses")
        parser.add_argument("--body-kwargs-file", type=str, default="default")
        parser.add_argument("--dataset-class", type=str, default="RigidBodyDataset")
        parser.add_argument("--n-train", type=int, default=800, help="number of train trajectories")
        parser.add_argument("--n-val", type=int, default=100, help="number of validation trajectories")
        parser.add_argument("--n-test", type=int, default=100, help="number of test trajectories")
        parser.add_argument("--is-reg-data", action="store_true", default=False)
        parser.add_argument("--is-lcp-data", action="store_true", default=False)
        parser.add_argument("--noise-std", type=float, default=0.0)
        # optimizer
        parser.add_argument("--chunk-len", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=200)
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--optimizer-class", type=str, default="AdamW")
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--SGDR", action="store_true", default=False)
        # model
        parser.add_argument("--hidden-size", type=int, default=256, help="number of hidden units")
        parser.add_argument("--num-layers", type=int, default=3, help="number of hidden layers")
        parser.add_argument("--tol", type=float, default=1e-7)
        parser.add_argument("--solver", type=str, default="rk4")
        parser.add_argument("--network-class", type=str, help="dynamical model",
                            choices=[
                                "CLNNwC", "CFLNN", "MLP_CD_CLNN", "IN_CP_SP", "IN_CP_CLNN"
                            ], default="CFLNN")

        parser.add_argument("--is-lcp-model", action="store_true", default=False)
        parser.add_argument("--is-reg-model", action="store_true", default=False)
        parser.add_argument("--reg", type=float, default=0.01)

        parser.add_argument("--hidden_dim", type=int, default=146)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--dropout_p", type=float, default=0.0)
        parser.add_argument("--embedding_dim", type=int, default=16)
        parser.add_argument("--horizon", type=int, default=5)
        parser.add_argument("--ODE_mode", type=str2bool, default=False)
        parser.add_argument("--fun_treatment", type=str2bool, default=False,
                            help="If true, the treatment response is an arbitrary function ( parametrized by a MLP )")
        parser.add_argument("--linear_output_fn", type=str2bool, default=False,
                            help="If true, the emission function is linear")
        parser.add_argument("--ipm_regul", type=str2bool, default=False, help="If true, uses IPM regularization")
        parser.add_argument("--alpha_reg", type=float, default=0.0,
                            help="Strength of alpha regularization. Only used if ipm_regul is True")
        parser.add_argument("--norm_encoding", type=str2bool, default=True,
                            help="If true, regularises the encoding. Only used if ipm_regul is True")
        parser.add_argument("--std_dev", type=str2bool, default=False,
                            help="If yes, stdandard deviations of the predictions are also returned and the model is trained with likelihood")
        parser.add_argument("--continuous_treatment_ode", type=str2bool, default=False,
                            help="If yes, uses an external ODE to model the treatments or an external function")
        parser.add_argument("--cf_var_regul", type=float, default=0.0, help="Counterfactual variance loss term")
        parser.add_argument("--MLP_decoding_mode", type=str2bool, default=False,
                            help="If true, uses an MLP with the times as input for the decoding")
        parser.add_argument("--sigma_sde", type=float, default=0.1, help="Diffusion parameter in the SDE prior")
        parser.add_argument("--output_scale", type=float, default=0.01,
                            help="standard deviation of the output_distribution")
        parser.add_argument("--kl_param", type=float, default=0.01,
                            help="beta parameter for the KL divergence term (theoretically 1. In practice , ....)")
        parser.add_argument("--start_scheduler", type=int, default=500,
                            help="iteration at which to start the linear scheduler for the KL term")
        parser.add_argument("--iter_scheduler", type=int, default=1000,
                            help="number of iterations in the the scheduler for the KL term")
        parser.add_argument("--ood_fact", type=float, default=0., help="Regularization for the OOD loss")
        parser.add_argument("--num_samples", type=int, default=3, help="number of sde samples to draw at each pass")
        parser.add_argument("--mc_dropout", type=float, default=0.,
                            help="MCDropout in the encoder. Dropout probability to apply")

        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()
    model = Model(hparams)

    is_reg_model = "_reg" if hparams.is_reg_model else ""
    is_lcp_model = "_lcp" if hparams.is_lcp_model else ""
    noise_std_str = "" if hparams.noise_std < 0.0000001 else f"_{hparams.noise_std}"
    savedir = os.path.join(".", "logs", 
                          hparams.body_kwargs_file + f"_{hparams.network_class}" 
                          + is_reg_model + is_lcp_model + f"_N{hparams.n_train}" + noise_std_str)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir, name='')

    checkpoint = ModelCheckpoint(monitor="val/mse",
                                 save_top_k=1,
                                 save_last=True,
                                 dirpath=tb_logger.log_dir
                                 )

    trainer = Trainer.from_argparse_args(hparams,
                                         deterministic=True,
                                         callbacks=[checkpoint],
                                         logger=[tb_logger],
                                         gpus=1,
                                         max_epochs=1000,
                                       #  resume_from_checkpoint=savedir+"/version_9/last.ckpt"
                                        )

    trainer.fit(model)
