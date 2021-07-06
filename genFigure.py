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

    if sys.argv[1] == "1":
        # Overlay Figure 1 AXL mutants diagram
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/AXLmuts_diagram.svg', 35, 0, scalee=0.06)

    if sys.argv[1] == "1":
        # Overlay Figure 1 Migration
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/Migration.svg', 550, 10, scalee=0.40)

    if sys.argv[1] == "1":
        # Overlay Figure 1 Island effect
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/island.svg', 765, 20, scalee=0.37)

    if sys.argv[1] == "2":
        # Overlay Figure 2 pipeline
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/pipeline.svg', 80, 10, scalee=0.15)

    if sys.argv[1] == "2":
        # Overlay Figure 2 heatmap
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/AXL_MS_heatmap.svg', 560, 0, scalee=0.09)

    if sys.argv[1] == 'M2':
        # Overlay Figure missingness cartoon
        overlayCartoon(fdir + 'figureM2.svg',
                       f'{cartoon_dir}/missingness_diagram.svg', 170, 0, scalee=1)

    if sys.argv[1] == "M5":
        # Overlay Figure tumor vs NATs heatmap
        overlayCartoon(fdir + 'figureM5.svg',
                       f'{cartoon_dir}/heatmap_NATvsTumor.svg', 50, 0, scalee=0.40)
