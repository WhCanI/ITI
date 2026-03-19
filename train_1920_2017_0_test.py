from sunpy.map import Map, all_coordinates_from_map
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from dateutil.parser import parse

from itipy.data import editor, dataset
from itipy.data.dataset import BaseDataset, KSOFlatDataset, KSODataset
from itipy.data.dataset import SDODataset, SOHODataset
from itipy.train.model import DiscriminatorMode
from itipy.trainer import Trainer
from itipy.callback import TBLossCallback, TBImageCallback
import warnings
from sunpy.util.exceptions import SunpyMetadataWarning
warnings.filterwarnings("ignore", category=SunpyMetadataWarning)

DEVICE = torch.device("cuda:1")
torch.cuda.set_device(DEVICE)

class Low1920(BaseDataset):
    def __init__(self, root, resolution=1024, ext='.fits', date_parser=None, **kw):
        editors = [
            editor.LoadMapEditor(),
            editor.FixDateEditor(),
            editor.KSOPrepEditor(),
            editor.NormalizeRadiusEditor(resolution, 0),
            editor.LimbDarkeningCorrectionEditor(),
            editor.MapToDataEditor(),
            editor.ImageNormalizeEditor(0.65, 1.5, stretch=editor.AsinhStretch(0.5)),
            editor.NanEditor(-1),
            editor.ReshapeEditor((1,resolution,resolution)),
        ]
        if date_parser is None:
            date_parser = lambda f: parse(os.path.basename(f)[3:-9])
        super().__init__(root, editors=editors, ext=ext, date_parser=date_parser, **kw)

def main():
    fits_1920_24_low_dir = r'/database2/jhpan/iti/low_fits'
    fits_2017_21_high_dir = r'/database2/jhpan/iti/high_fits'
    base_dir = "./low_to_high_kso"
    tb_dir = os.path.join(base_dir, "tb")

    os.makedirs(base_dir, exist_ok=True)

    trainer = Trainer(
        input_dim_a          = 1,
        input_dim_b          = 1,
        upsampling           = 0,
        discriminator_mode   = DiscriminatorMode.SINGLE,
        lambda_diversity     = 1,
        lambda_content       = 5,
        n_filters=48,
        norm                 = 'in_rs_aff'
    )

    low_1920_train = Low1920(fits_1920_24_low_dir, resolution=512, months=list(range(11)))
    high_2017_train = KSOFlatDataset(fits_2017_21_high_dir, resolution=512, months=list(range(11)))

    low_1920_val = Low1920(fits_1920_24_low_dir, resolution=512, months=[11, 12])
    high_2017_val = KSOFlatDataset(fits_2017_21_high_dir, resolution=512, months=[11, 12])

    trainer.startBasicTraining(
        base_dir = base_dir,
        ds_A = low_1920_train,
        ds_B = high_2017_train,
        ds_valid_A = low_1920_val,
        ds_valid_B = high_2017_val,
        iterations= 180_000,
        num_workers= 4,
        batch_size= 1,
        validation_history = True,
        additional_callbacks = [
                TBLossCallback(trainer, tb_dir, log_every=100),
                TBImageCallback(trainer, low_1920_val, high_2017_val,
                                period=2000, logdir=os.path.join(tb_dir, "images"))
            ]
    )

if __name__ == '__main__':
    main()

