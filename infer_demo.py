import torch, matplotlib.pyplot as plt
from itipy.data.dataset import BaseDataset
import os
from dateutil.parser import parse
from itipy.data import editor, dataset

DEVICE = torch.device("cuda:0")
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


GEN_PATH   = "./low_to_high_kso/generator_AB.pt"
LOW_DATA   = "/home/jhpan/ITI/InstrumentToInstrument/dataset/1920_fits"

infer_files = [r"/database2/jhpan/iti/low_fits/HA_19200202T081800_cal.fits",
               r"/database2/jhpan/iti/low_fits/HA_19210823T080700_cal.fits",
               r"/database2/jhpan/iti/low_fits/HA_19220519T085800_cal.fits",
               r"/database2/jhpan/iti/low_fits/HA_19230130T102200_cal.fits",
               r"/database2/jhpan/iti/low_fits/HA_19241005T082500_cal.fits"]

gen_ab = torch.load(GEN_PATH, map_location="cuda")
gen_ab.eval()

low_ds = Low1920(infer_files, resolution=512)
idxs   = [0, 1, 2, 3, 4]

with torch.no_grad():
    for i in idxs:
        x = torch.from_numpy(low_ds[i]).unsqueeze(0).cuda().float()  # [1,C,H,W]
        y = gen_ab(x)  # infer

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Low-1920")
        plt.imshow(x[0, 0].cpu(), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("High")
        plt.imshow(y[0, 0].cpu(), cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"./low_to_high_kso/sample_{i}_i.png", dpi=150)

        plt.show()
