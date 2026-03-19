import torch, matplotlib.pyplot as plt, os
from dateutil.parser import parse

from itipy.data import editor
from itipy.data.dataset import BaseDataset, KSOFlatDataset

class Low1920(BaseDataset):
    def __init__(self, root, resolution=512, ext=".fits", date_parser=None, **kw):
        editors_ = [
            editor.LoadMapEditor(),
            editor.FixDateEditor(),
            editor.KSOPrepEditor(),
            editor.NormalizeRadiusEditor(resolution, 0),
            editor.LimbDarkeningCorrectionEditor(),
            editor.MapToDataEditor(),
            editor.ImageNormalizeEditor(0.65, 1.5, stretch=editor.AsinhStretch(0.5)),
            editor.NanEditor(-1),
            editor.ReshapeEditor((1, resolution, resolution)),
        ]
        if date_parser is None:
            date_parser = lambda f: parse(os.path.basename(f)[3:-9])
        super().__init__(root, editors=editors_, ext=ext, date_parser=date_parser, **kw)


GEN_BA = "/home/jhpan/ITI/InstrumentToInstrument/low_to_high_kso/generator_BA.pt"
HIGH_DIR = "/database2/jhpan/iti/high_fits"

gen_ba = torch.load(GEN_BA, map_location="cuda")
gen_ba.eval()

high_ds = KSOFlatDataset(HIGH_DIR, resolution=512, limit=4)
idxs    = [56, 89, 123, 985]

noise_dim     = 16
depth_noise   = 4
upsampling    = 0
spatial_scale = 512 // 2**(depth_noise + upsampling)

with torch.no_grad():
    for i in idxs:
        x = torch.from_numpy(high_ds[i]).unsqueeze(0).cuda().float()   # [1,C,H,W]

        n = torch.rand(1, noise_dim, spatial_scale, spatial_scale, device="cuda")

        y = gen_ba(x, n)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.title("High"); plt.imshow(x[0, 0].cpu(), cmap="gray"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Low"); plt.imshow(y[0, 0].cpu(), cmap="gray"); plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"./low_to_high_kso/ba_sample_{i}.png", dpi=150)
        plt.show()
