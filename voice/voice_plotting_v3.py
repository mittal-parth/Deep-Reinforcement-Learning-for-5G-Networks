#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import scipy.special
import matplotlib.ticker as tick

import os

import tikzplotlib
import pandas as pd


def generate_ccdf(data0, data1, data2, data3):
    fig = plt.figure(figsize=(10.24,7.68))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

    num_bins = 50
    for data in [data0, data1, data2, data3]:

        data_ = data

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        ccdf = 1 - np.cumsum(counts) / counts.sum()
        ccdf = np.insert(ccdf, 0, 1)
        bin_edges = np.insert(bin_edges[1:], 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
        ax = fig.gca()
        ax.plot(bin_edges, ccdf)

    labels = ['Optimal', 'Deep $Q$-learning (proposed)', 'Tabular $Q$-learning', 'Fixed Power Allocation (FPA)']
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')
    ax.set_ylim([0,1])
    ax.legend(labels, loc="best")
    plt.grid(True)
    # plt.tight_layout()

    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    print(script_path)
    # Get the directory containing the script
    script_directory = os.path.dirname(script_path)
    print(script_directory)
    # Combine script directory with the filenames
    pdf_filepath = os.path.join(script_directory, 'voice_ccdf.pdf')
    tikz_filepath = os.path.join(script_directory, 'voice_ccdf.tikz')

    print(pdf_filepath)
    print(tikz_filepath)

    plt.savefig(pdf_filepath, format="pdf")
    tikzplotlib.save(tikz_filepath)
    plt.close(fig)

def read_output(filename):
    df = pd.read_csv(filename)
    df = df.T
    df = df.reset_index()
    df.drop(df.index[-1], axis=0, inplace=True)
    df = df.astype(float)

    return list(df.iloc[:,0])

def main():

    # TODO, put all reported SINRs for both UEs in a vector
    ue_deep_ue1 = read_output('figures_deep\\ue_1_sinr.txt')
    ue_tabular_ue1 = read_output('figures_tabular\\ue_1_sinr.txt')
    ue_fpa_ue1 = read_output('figures_fpa\\ue_1_sinr.txt')
    ue_opt_ue1 = read_output('figures_optimal\\ue_1_sinr.txt')

    ue_deep_ue2 = read_output('figures_deep\\ue_2_sinr.txt')
    ue_tabular_ue2 = read_output('figures_tabular\\ue_2_sinr.txt')
    ue_fpa_ue2 = read_output('figures_fpa\\ue_2_sinr.txt')
    ue_opt_ue2 = read_output('figures_optimal\\ue_2_sinr.txt')

    ue_deep = np.array(ue_deep_ue1+ue_deep_ue2)
    ue_tabular = np.array(ue_tabular_ue1+ue_tabular_ue2)
    ue_fpa = np.array(ue_fpa_ue1+ue_fpa_ue2)
    ue_opt = np.array(ue_opt_ue1+ue_opt_ue2)

    generate_ccdf(ue_opt, ue_deep, ue_tabular, ue_fpa)

if __name__ == '__main__':
    main()



