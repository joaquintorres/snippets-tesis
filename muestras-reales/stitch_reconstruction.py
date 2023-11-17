import numpy as np

import fpmsample.math as pmath
from stitch import Patch, Picture
from fpmsample.simicro import RealMicroscope
from phaseopt.containers import Options, ResultsContainer
from phaseopt.phaseopt import Solver

import itertools as it

dirname = "filename.h5py"

rm = RealMicroscope.from_dirname(dirname=dirname, color="B")
overlap = 15
pic_hr = Picture.from_microscope(rm, overlap=overlap, outfile="ft_image.h5py")

opts = Options(
    record_rss=True,
    record_angular_rss=True,
    xt=None,
    lazy=False,
    max_iter=7,
)
results = ResultsContainer(opts)

for patch in pic_hr.patches():
    pupil = pmath.aberrated_pupil(0e-6, rm)
    rm.y0, rm.x0 = pic_hr.get_lr_center(patch.position)
    delta_gk = rm.get_phaseopt_input(background_matrix=10000)
    centers = rm.centers(mode="raw")
    centers = np.array((list(centers)))
    n = 16
    delta_gk = it.islice(delta_gk, n)
    centers = centers[:n]
    solver = Solver(
        delta_gk,
        centers,
        rm.no,
        rm.patch,
        rm.nsyn,
        results=results,
        p0=pupil,
        microscope=rm,
        method="known_corrections",
    )
    res = solver.run()
 
    delta_gk = rm.get_phaseopt_input(background_matrix=1000)
    patch.image = np.flip(res.gk, axis=0)

pic_hr.store_array()
complete_image = pic_hr.build_complete_array()

