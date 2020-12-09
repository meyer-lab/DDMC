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

    if sys.argv[1] == 'M2':
        # Overlay Figure M2 cartoon
        overlayCartoon(fdir + 'figureM2.svg',
                       f'{cartoon_dir}/missingness_diagram.svg', 0, 0, scalee=0.13)

    if sys.argv[1] == "M3":
        # Overlay Figure M3 heatmap
        overlayCartoon(fdir + 'figureM3.svg',
                        f'{cartoon_dir}/heatmap_fM3.svg', 50, 25, scalee=0.35)
