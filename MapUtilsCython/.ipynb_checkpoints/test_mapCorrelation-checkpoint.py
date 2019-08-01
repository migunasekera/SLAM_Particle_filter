#!/usr/bin/env python3

import numpy as np
import MapUtils_fclad as mu
import MapUtils_original as muo
from scipy import io
import sys
import time


if __name__ == "__main__":

    dataIn = io.loadmat("../data/Hokuyo20.mat")
    speedup = np.zeros(10)
    for i in range(0, 10000, 1000):
        print(i)
        angles = np.double(dataIn['Hokuyo0']['angles'][0][0])
        ranges = np.array([dataIn['Hokuyo0']['ranges'][0][0][:,0]]).T

        # take valid indices
        indValid = np.logical_and((ranges < 30), (ranges > 0.1))
        ranges = ranges[indValid]
        angles = angles[indValid]

        # init MAP
        MAP = {}
        MAP['res'] = 0.05  # meters
        MAP['xmin'] = -20  # meters
        MAP['ymin'] = -20
        MAP['xmax'] = 20
        MAP['ymax'] = 20
        MAP['sizex'] = int(np.ceil(
            (MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # cells
        MAP['sizey'] = int(np.ceil(
            (MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)

        # xy position in the sensor frame
        xs0 = np.array([ranges*np.cos(angles)])
        ys0 = np.array([ranges*np.sin(angles)])

        # convert position in the map frame here
        Y = np.concatenate([np.concatenate([xs0, ys0], axis=0),
                            np.zeros(xs0.shape)], axis=0)
        # convert from meters to cells
        xis = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16)-1
        yis = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16)-1

        indGood = np.logical_and(
            np.logical_and(
                np.logical_and((xis > 1),
                               (yis > 1)),
                (xis < MAP['sizex'])),
            (yis < MAP['sizey']))
        MAP['map'][xis[0][indGood[0]], yis[0][indGood[0]]] = 1

        x_im = np.arange(MAP['xmin'], MAP['xmax']+MAP['res'], MAP['res'])
        y_im = np.arange(MAP['ymin'], MAP['ymax']+MAP['res'], MAP['res'])

        x_range = np.arange(-0.2, 0.2+0.005, 0.005) + np.random.rand(1)*5
        y_range = np.arange(-0.2, 0.2+0.005, 0.005) + np.random.rand(1)*5

        print("Testing map_correlation...")
        start = time.time()
        c = mu.mapCorrelation_fclad(MAP['map'], x_im, y_im,
                                    Y[0:3, :],
                                    x_range, y_range)
        end1 = time.time()
        c2 = muo.mapCorrelation(MAP['map'], x_im, y_im,
                                Y[0:3, :],
                                x_range, y_range)
        end2 = time.time()
        if np.sum(c == c2) == np.size(c):
            print("...Test passed.")
            speedup[i//1000] = (end1-start)/(end2-end1)
        else:
            print("...Test failed.")
            sys.exit(1)
    print(np.average(speedup))
