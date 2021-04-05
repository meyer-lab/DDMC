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
                       f'{cartoon_dir}/AXLmuts_diagram.svg', 10, 23, scalee=0.26)

    # if sys.argv[1] == "1":
    #     # Overlay Figure 1 Migration
    #     overlayCartoon(fdir + 'figure1.svg',
    #                    f'{cartoon_dir}/Migration.svg', 550, 10, scalee=0.40)

    if sys.argv[1] == "1":
        # Overlay Figure 1 Island effect
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/island.svg', 765, 20, scalee=0.37)

    if sys.argv[1] == "2":
        # Overlay Figure 2 heatmap
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/AXL_MS_heatmap.svg', 50, 10, scalee=0.11)

    if sys.argv[1] == "2":
        # Overlay Figure 2 KSEA plot
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/KSEA_WTvsKO.svg', 90, 520, scalee=0.27)

    if sys.argv[1] == "3":
        # Overlay Figure 3 computational pipeline
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/pipeline.svg', 175, 0, scalee=0.01)

    if sys.argv[1] == "4":
        # Overlay Figure 4 YAP AXL vs KO blot
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/YAP_WTvsKO_blot.svg', 700, 590, scalee=0.17)

    if sys.argv[1] == "4":
        # Overlay Figure 4 YAP dasatinib dose response
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/YAP_DR_blot.svg', 50, 600, scalee=0.17)

    if sys.argv[1] == "S6":
        # Overlay Figure 3 cross validation diagram
        overlayCartoon(fdir + 'figureS6.svg',
                       f'{cartoon_dir}/cross_validation.svg', 60, 0, scalee=0.21)

    if sys.argv[1] == "M3":
        # Overlay Figure M3 heatmap
        overlayCartoon(fdir + 'figureM3.svg',
                       f'{cartoon_dir}/heatmap_fM3.svg', 110, 10, scalee=0.32)

    if sys.argv[1] == 'MS6':
        # Overlay Figure MS6 cartoon
        overlayCartoon(fdir + 'figureMS6.svg',
                       f'{cartoon_dir}/missingness_diagram.svg', 0, 0, scalee=0.13)



