#!/usr/bin/env python3
from msresist.figures.common import overlayCartoon
import sys
import logging
import time
import matplotlib

matplotlib.use("AGG")

cartoon_dir = r"./msresist/figures"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if __name__ == "__main__":
    nameOut = "figure" + sys.argv[1]

    if "M" in nameOut:
        fdir = "./output/method/"
    else:
        fdir = "./output/biol/"

    start = time.time()

    exec("from msresist.figures." + nameOut + " import makeFigure")
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

    logging.info("%s is done after %s seconds.", nameOut, time.time() - start)

    if sys.argv[1] == 'M2':
        # Overlay Figure missingness cartoon
        overlayCartoon(fdir + 'figureM2.svg',
                       f'{cartoon_dir}/missingness_diagram.svg', 75, 5, scalee=1.1)

    if sys.argv[1] == "M5":
        # Overlay Figure tumor vs NATs heatmap
        overlayCartoon(fdir + 'figureM5.svg',
                       f'{cartoon_dir}/heatmap_NATvsTumor.svg', 50, 0, scalee=0.40)
