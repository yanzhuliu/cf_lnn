import os, sys
THIS_DIR = os.path.abspath('')
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)
import csv
import json

from pytorch_lightning import seed_everything
import torch

from trainer import Model, str_to_class
from systems.bouncing_point_masses import BouncingPointMasses
from systems.chain_pendulum_with_contact import ChainPendulumWithContact

seed_everything(0)

model_names = [
    "CFLNN",
]
system_names = [
    "_BP5_cf_1",
#    "_BP5_cf_2",
#    "_BP5_cf_3"
#    "_CP3_cf_3",
]
class_names = [
    "BouncingPointMasses",
 #   "BouncingPointMasses",
 #   "BouncingPointMasses"
    # "ChainPendulumWithContact",
#    "ChainPendulumWithContact",
]

bodies = {}
for i in range(len(class_names)):
    with open(os.path.join(THIS_DIR, "examples", system_names[i] +".json"), "r") as file:
        body_kwargs = json.load(file)
    bodies[system_names[i]] = str_to_class(class_names[i])(system_names[i], **body_kwargs)

#%%


results = {}
for system_name in system_names:
    results[system_name] = {}
    for model_name in model_names:
        print(f"calculating {system_name}, {model_name}")
        checkpoint_path = os.path.join(
            THIS_DIR,
            "logs",
            f"{system_name}_{model_name}_N800",
            "version_0",
            "best.ckpt"
        )
        model = Model.load_from_checkpoint(checkpoint_path)

        model.hparams.batch_size = 100
        model.eval()
        dataloader = model.test_dataloader()
        test_batch = next(iter(dataloader))
        with torch.no_grad():
            results[system_name][model_name] = model.test_step(
                test_batch,
                0,
                50 * model.hparams.dt,
            )

for system_name in system_names:
    fa = open(f"test_{system_name}_abs_err_.csv", 'w')
    writerA = csv.writer(fa)
    fd = open(f"test_{system_name}_mape_err_.csv", 'w')
    writerD = csv.writer(fd)
    A = []
    D = []
    header = []
    for model_name in model_names:
        A.append(results[system_name][model_name]["abs_err"].mean(0).tolist())
        D.append(results[system_name][model_name]["mape_err"].mean(0).tolist())
        header.append(f"{model_name}")
    writerA.writerow(header)
    writerA.writerows(zip(*A))
    writerD.writerow(header)
    writerD.writerows(zip(*D))
    fa.close()
    fd.close()
