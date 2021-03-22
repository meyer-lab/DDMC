#!/usr/bin/env python3
from msresist.figures.common import overlayCartoon
import sys
import logging
import time
import matplotlib

matplotlib.use("AGG")

fdir = "./"
cartoon_dir = r"./msresist/figures"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if __name__ == "__main__":
    nameOut = "figure" + sys.argv[1]

    start = time.time()

    exec("from msresist.figures." + nameOut + " import makeFigure")
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

    logging.info("%s is done after %s seconds.", nameOut, time.time() - start)

    if sys.argv[1] == 'MS6':
        # Overlay Figure MS6 cartoon
        overlayCartoon(fdir + 'figureMS6.svg',
                       f'{cartoon_dir}/missingness_diagram.svg', 0, 0, scalee=0.13)

    if sys.argv[1] == "M3":
        # Overlay Figure M3 heatmap
        overlayCartoon(fdir + 'figureM3.svg',
                       f'{cartoon_dir}/heatmap_fM3.svg', 110, 10, scalee=0.32)

    if sys.argv[1] == "2":
        # Overlay Figure 1 heatmap
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/AXL_MS_heatmap.svg', 40, 10, scalee=0.40)

    if sys.argv[1] == "3":
        # Overlay Figure 3 computational pipeline
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/pipeline.svg', 250, 0, scalee=0.22)

    if sys.argv[1] == "3":
        # Overlay Figure 3 cross validation diagram
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/cross_validation.svg', 850, 17, scalee=0.17)
