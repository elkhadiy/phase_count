#!/usr/bin/env python3

import os
import sys
import argparse
#import logging
import json
import numpy as np
import sympy as sp
from scipy.spatial.distance import sqeuclidean
import matplotlib.pyplot as plt

def extract_phases(
        popsi_file_name,
        radius, width, delta_y, delta_z,
        number_samples_y, number_samples_z, snap
    ):

    popsi = open(popsi_file_name, "rb")

    # extract phase data from P_Opsi file
    popsi.seek(number_samples_y * number_samples_z * 16 * snap)
    frame = np.fromfile(
        popsi,
        dtype=np.float64,
        count=number_samples_y * number_samples_z * 2
    )
    rel = frame[0::2]
    img = frame[1::2]
    phase = np.angle(rel + img * 1j)
    grid = phase.reshape((number_samples_y, number_samples_z))

    # convert point coordinates to matrix row and column indexes
    z2j = lambda z: int(
        z / delta_z + (number_samples_z + 1) / 2
    )
    y2i = lambda y_axis_samples: int(
        y_axis_samples / delta_y + (number_samples_y + 1) / 2
    )

    # extracts the phase circles
    #   circle = -1 : the left one
    #   circle = +1 : the right one
    get = lambda r, theta, circle: grid[
        y2i(0 + r * np.sin(theta)),
        z2j(circle * radius + r * np.cos(theta))
    ]

    # discretisation steps for extracting the phase from the simulation
    number_samples_theta = 500
    number_samples_r = 100

    theta_samples = np.linspace(0, 2 * np.pi, number_samples_theta)

    ribbon = [np.array(
        [
            [get(r, theta, i) for theta in theta_samples]
            for r in np.linspace(
                radius - width,
                radius + width,
                number_samples_r
            )
        ]
    ) for i in [-1, 1]]

    return ribbon

def gen_sawtooth(winding_number, number_samples_theta):
    ''' winding_number = +- slope '''

    x_axis_samples = np.linspace(0, 2 * np.pi, number_samples_theta)
    y_axis_samples = np.zeros(number_samples_theta)

    slope_func = lambda x: winding_number * x - np.sign(winding_number) * np.pi

    if winding_number:
        x_slope_samples = int(number_samples_theta / abs(winding_number))
    else:
        return y_axis_samples

    for i in range(abs(winding_number)):
        for i0 in range(x_slope_samples):
            y_axis_samples[i * x_slope_samples + i0] = slope_func(x_axis_samples[i + i0])

    return y_axis_samples

def find_winding_number(ribbon):
    '''
    find the winding numbers that minimizes the distance
    between the phase distribution and its theoretical counter-part which is
    the sawtooth function
    '''
    number_samples_theta = ribbon[0].shape[1]

    y_axis_samples = [np.roll(
        ribbon[i][int(len(ribbon[i])/2)],
        len(ribbon[i][int(len(ribbon[i])/2)])
        - ribbon[i][int(len(ribbon[i])/2)].argmin()
    ) for i in [0, 1]]

    def distance(winding_number, circle):
        return sqeuclidean(
                    gen_sawtooth(winding_number, number_samples_theta),
                    y_axis_samples[circle]
                )

    dl = np.zeros(201)
    dr = np.zeros(201)
    for i in range(-100, 101, 1):
        dl[i + 100] = distance(i, 0)
        dr[i + 100] = distance(i, 1)
    
    wl = np.argmin(dl) - 100
    wr = np.argmin(dr) - 100

    return wl, wr

def plot_phases(ribbon):

    number_samples_theta = ribbon[0].shape[1]

    x_axis_samples = np.linspace(0, 2 * np.pi, number_samples_theta)

    y_axis_samples = [np.roll(
        ribbon[i][int(len(ribbon[i])/2)],
        len(ribbon[i][int(len(ribbon[i])/2)])
        - ribbon[i][int(len(ribbon[i])/2)].argmin()
    ) for i in [0, 1]]

    # plot the phase for the two circles

    plt.style.use("fivethirtyeight")

    fig, axs = plt.subplots(2, 1, figsize=(16, 8))

    detail_pis = np.array([0, sp.pi / 6, sp.pi / 4, sp.pi / 3, sp.pi / 2])
    minimal_pis = np.array([0, sp.pi / 4, sp.pi / 2])

    zero2pi = np.unique(
        np.array([detail_pis + i * sp.pi / 2 for i in range(4)]).flatten()
    )
    npi2pi = np.unique(
        np.array(
            [minimal_pis + i * sp.pi / 2 for i in range(-2, 2)]
        ).flatten()
    )

    s2tex = np.vectorize(
        lambda x_axis_samples: '$'+sp.latex(x_axis_samples)+'$'
    )

    axis_lash = 0.2
    for axis in axs:
        axis.set(
            xlim=[0 - axis_lash, 2 * np.pi + axis_lash],
            ylim=[-np.pi - axis_lash, np.pi + axis_lash],
            xticks=zero2pi.astype(float),
            yticks=npi2pi.astype(float),
            xticklabels=s2tex(zero2pi),
            yticklabels=s2tex(npi2pi)
        )
        axis.spines['left'].set_position('zero')
        axis.spines['bottom'].set_position('zero')
        axis.spines['left'].set_color('#adaca7aa')
        axis.spines['bottom'].set_color('#adaca7aa')

    circle = ['Left', 'Right']

    winding_number = find_winding_number(ribbon)

    for i, axis in enumerate(axs):
        axis.plot(x_axis_samples, y_axis_samples[i], alpha=0.3, color='b')
        axis.axhline(
            min(y_axis_samples[i]),
            linestyle='-.', color='g', alpha=0.3
        )
        axis.axhline(
            max(y_axis_samples[i]),
            linestyle='-.', color='g', alpha=0.3
        )
        axis.plot(
            x_axis_samples,
            gen_sawtooth(winding_number[i], number_samples_theta),
            alpha=0.3, color='r'
        )
        axis.fill_between(
            x_axis_samples, y_axis_samples[i],
            gen_sawtooth(winding_number[i], number_samples_theta),
            alpha=0.2, color='r'
        )
        axis.set_title(circle[i] + " Circle : " + str(winding_number[i]))

    result = {
        'Left circle': int(winding_number[0]),
        'Right circle': int(winding_number[1])
    }

    return fig, axs, result

# gather program code in a main() function
def main(args):

    # basic check for the P_Opsi file
    if not os.path.isfile(args.P_Opsi):
        print(args.P_Opsi + " not a valid file name")
        sys.exit(-1)

    ribbon = extract_phases(
        popsi_file_name=args.P_Opsi,
        radius=5.0,
        width=1.2,
        delta_y=0.07, delta_z=0.07,
        number_samples_y=280, number_samples_z=450,
        snap=1000
    )

    fig, axs, result = plot_phases(ribbon)

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f)

# standard boilerplate for a commandline python script
# calls the main function and sets up the commandline arguments
if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(
        description="Plots phase shift from P_Opsi.dat file.",
        epilog="As an alternative to the commandline, "
               "params can be placed in a file, one per line, "
               "and specified on the commandline like "
               "'%(prog)s @params.conf'.",
        fromfile_prefix_chars='@'
    )

    PARSER.add_argument(
        "P_Opsi",
        help="simulation file called FILE from which to extract phase data",
        metavar="FILE"
    )

    PARSER.add_argument(
        "-s",
        "--save",
        type=str,
        help="save figure to a file named FIGURE",
        metavar="FIGURE"
    )

    PARSER.add_argument(
        "-o",
        "--output",
        type=str,
        help="save winding number result to a file named RESULT",
        metavar="RESULT"
    )

    ARGS = PARSER.parse_args()

    main(ARGS)
