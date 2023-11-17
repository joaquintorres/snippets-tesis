#!/usr/bin/env python3
from fpmsample.simicro import SimMicroscope
import phaseopt.data as data
from phaseopt.containers import Options, ResultsContainer
from phaseopt import Solver
import numpy as np
import fpmsample.math as pmm

import argparse
from matplotlib.backends.backend_pdf import PdfPages
from rich.progress import track
import cv2 
from scipy.fftpack import fft2
from matplotlib.colors import LogNorm
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg') #don't instantiate windows that don't get displayed, since that causes problems
# Arguments
parser = argparse.ArgumentParser(description='Cycle through different kernel sizes for gaussian blurs')
parser.add_argument('library_index', type=int)
parser.add_argument('--filename', type=str, default="test.pdf")
parser.add_argument('--fft', type=bool, default=False)
args = parser.parse_args()

def plot_phase(img, axs=None, title = ""):
    if axs == None:
        raise Exception("Empty Axis")
    imshw = axs.imshow(img, vmin = -np.pi, vmax = np.pi)
    plt.colorbar(imshw, ax = axs)
    axs.set_title(title)

def plot_mag(img, axs=None, title = ""):
    if axs == None:
        raise Exception("Empty Axis")
    imshw = axs.imshow(img, vmin = 0, vmax = 100000)
    plt.colorbar(imshw, ax = axs)
    axs.set_title(title)

#Horrible, when done correctly causes core dump
def plot_fft(img, axs=None, title = ""):
    if axs == None:
        raise Exception("Empty Axis")
    imshw = axs.imshow(img, norm = LogNorm(vmin=5))
    plt.colorbar(imshw, ax = axs)
    axs.set_title(title)
# Reconstruction
def reconstruct(cfg, index):
    sm = SimMicroscope(cfg=cfg, image_index=index, load_image_mode='sequential')
    sm.generate_samples()
    sm.details()
    delta_gk, centers = sm.get_phaseopt_input()

    opts = Options(record_rss=True, record_angular_rss=True, delta=1E-12,
                xt=sm.complex_image, max_iter=15, lazy=False)

    results = ResultsContainer(opts)

    solver = Solver(delta_gk, centers, sm.no, sm.patch, sm.nsyn, results=results,
                    microscope=sm, method='error_reduction')

    res = solver.run()
    return res, sm 

cfg = data.cfg_load()
res, sm = reconstruct(cfg, args.library_index)
reconstruction = pmm.rotate_match(res.gk, sm.complex_image)
rec_mag = np.abs(reconstruction)
rec_phase = np.angle(reconstruction)

mag = np.abs(sm.complex_image)
phase = np.angle(sm.complex_image)
fft_mag = fft2(mag)
fft_phase = fft2(phase)

dims = np.arange(2,21) #tested kernel sizes
with PdfPages(args.filename) as pdf:
    for dim in track(dims):
        blur_mag = cv2.blur(mag,(dim,dim))
        blur_phase = cv2.blur(phase,(dim,dim))
        fft_blur_mag = fft2(blur_mag)
        fft_blur_phase = fft2(blur_phase)

        fig, ax = plt.subplots(2,2)
        # Image diffs
        plot_mag(blur_mag, axs = ax[0,0], title = "Magnitude kernel {}".format((dim,dim)))
        plot_mag(np.abs(blur_mag - rec_mag), axs = ax[0,1], title = "Magnitude diff {}".format((dim,dim)))
        plot_phase(blur_phase, axs = ax[1,0], title = "Phase kernel {}".format((dim,dim)))
        plot_phase(np.abs(blur_phase - rec_phase), axs = ax[1,1], title = "Phase diff {}".format((dim,dim)))
        pdf.savefig(fig)
        plt.close()
        if args.fft:
            fig, ax = plt.subplots(2,2)
            # FFT diffs
            plot_fft(np.abs(fft_blur_mag), axs = ax[0,0], title = "Magnitude kernel {}".format((dim,dim)))
            plot_fft(np.abs(fft_blur_mag - fft_mag), axs = ax[0,1], title = "Magnitude diff {}".format((dim,dim)))
            plot_fft(np.abs(fft_blur_phase), axs = ax[1,0], title = "Phase kernel {}".format((dim,dim)))
            plot_fft(np.abs(fft_blur_phase - fft_phase), axs = ax[1,1], title = "Phase diff {}".format((dim,dim)))
            pdf.savefig(fig)
            plt.close()
