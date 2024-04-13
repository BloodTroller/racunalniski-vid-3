import math
import sys
import time
import warnings

import numba
from numba import jit
import cv2 as cv
import numpy as np

def kmeans(slika, og_centers, num, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    if k < 1 or iteracije < 1:
        return slika

    output = slika.copy()
    vis = output.shape[0]
    sir = output.shape[1]
    centers = og_centers.copy()
    data_size = len(centers[0])
    height, width = slika.shape[:2]
    #dtype=numba.int64
    cores = np.zeros((width, height), dtype=int)

    for it in range(iteracije):
        divider = [0] * num
        # dtype=numba.int64
        nums = np.zeros((data_size, num), dtype=int)
        for x in range(width):
            for y in range(height):
                h, s, v = slika[x, y]
                centre_index = -1
                previous_distance = sys.maxsize
                for i in range(num):
                    h2, s2, v2 = centers[i]
                    distance = math.sqrt((h2 - h) ** 2 + (s2 - s) ** 2 + (v2 - v) ** 2)
                    if distance < previous_distance:
                        centre_index = i
                        previous_distance = distance
                closest_centre = centers[centre_index]
                # print("\nClosest centre for pixel " + str(x+y) + ":" + str(closest_centre))
                cores[x, y] = centre_index
                nums[centre_index][0] += h
                nums[centre_index][1] += s
                nums[centre_index][2] += v
                divider[centre_index] += 1

        for j in range(num):
            if divider[j] > 0:
                nums[j][0] /= divider[j]
                nums[j][1] /= divider[j]
                nums[j][2] /= divider[j]

    for xx in range(width):
        for yy in range(height):
            output[xx, yy] = centers[cores[xx, yy]]

    return output

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass


def izracunaj_centre(slika, n=1, manual=True, t=5, big_array=False):
    '''Izračuna centre za metodo kmeans.'''
    centre = []
    if manual:
        cv.namedWindow("centres")
        a = -1
        b = -1
        for st in range(n):
            click = False
            title = "Choose " + str(st + 1) + ". centre"
            cv.setWindowTitle("centres", title)

            def click_event(event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN:
                    nonlocal a
                    nonlocal b
                    nonlocal title
                    a = x
                    b = y
                    cv.setWindowTitle("centres", title + " (" + str(x) + ", " + str(y) + ")")

            cv.imshow("centres", slika)
            cv.setMouseCallback("centres", click_event)
            cv.waitKey(0)
            h, s, v = slika[a, b]
            if big_array:
                centre.append([h, s, v, a, b])
            else:
                centre.append([h, s, v])

        cv.destroyWindow("centres")
    else:
        pass
    return centre


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    k = 3
    iteracije = 10
    slika = cv.imread(".utils/lenna.png")
    cv.cvtColor(slika, cv.COLOR_BGR2HSV)

    # slika = cv.resize(slika, (32, 32))

    segmentirana_slika = kmeans(slika, izracunaj_centre(slika, 3), 3, k, iteracije)

    cv.setWindowTitle("slika", "Normalna slika")
    cv.imshow("slika", slika)
    cv.waitKey(0)

    cv.setWindowTitle("slika", "Slika obdealana s KMEANS algoritmom")
    cv.imshow("slika", segmentirana_slika)
    cv.waitKey(0)

    cv.destroyAllWindows()
    pass
