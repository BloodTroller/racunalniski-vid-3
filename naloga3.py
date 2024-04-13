import math
import random
import sys
import time
import warnings

import numba
from numba import jit
import cv2 as cv
import numpy as np


def dist(color1, color2):
    difs = []

    for i in range(len(color1)):
        c1 = color1[i]
        c2 = color2[i]
        squared_difference = (c1 - c2) ** 2
        difs.append(squared_difference)

    distance = math.sqrt(sum(difs))
    return distance

def kmeans(slika, centers, num=3, iteracije=10):
    print(centers)
    print("\n")
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    if k < 1 or iteracije < 1:
        return slika

    output = slika.copy()
    data_size = len(centers[0])
    if data_size == 5:
        is_big = True
    elif data_size == 3:
        is_big = False
    else:
        raise Exception("Data size is incorrect! (should be 5 or 3)!")

    height, width = slika.shape[:2]
    # dtype=numba.int64
    cores = np.zeros((width, height), dtype=int)

    for it in range(iteracije):
        print("Iteracija: " + str(it + 1) + "\n")
        divider = [0] * num
        # dtype=numba.int64
        nums = np.zeros((num, data_size), dtype=int)
        for x in range(width):
            for y in range(height):
                h, s, v = slika[y, x]
                centre_index = -1
                previous_distance = sys.maxsize
                for i in range(num):
                    if is_big:
                        h2, s2, v2, x2, y2 = centers[i]
                    else:
                        h2, s2, v2 = centers[i]
                    distance = (h2 - h) ** 2 + (s2 - s) ** 2 + (v2 - v) ** 2

                    if is_big:
                        distance += (x2 - x) ** 2 + (y2 - y) ** 2

                    distance = math.sqrt(distance)

                    if distance < previous_distance:
                        centre_index = i
                        previous_distance = distance
                closest_centre = centers[centre_index]
                # print("\nClosest centre for pixel " + str(x+y) + ":" + str(closest_centre))
                cores[x, y] = centre_index
                nums[centre_index][0] += h
                nums[centre_index][1] += s
                nums[centre_index][2] += v
                if is_big:
                    nums[centre_index][3] += x
                    nums[centre_index][4] += y
                divider[centre_index] += 1

        for j in range(num):
            if divider[j] > 0:
                for stevnik in range(data_size):
                    nums[j][stevnik] /= divider[j]
                mh = nums[j][0]
                ms = nums[j][1]
                mv = nums[j][2]
                if is_big:
                    mx = nums[j][3]
                    my = nums[j][4]
                    centers[j] = (mh, ms, mv, mx, my)
                else:
                    centers[j] = (mh, ms, mv)

    for xx in range(width):
        for yy in range(height):
            output[yy, xx][0] = centers[cores[xx, yy]][0]
            output[yy, xx][1] = centers[cores[xx, yy]][1]
            output[yy, xx][2] = centers[cores[xx, yy]][2]

    return output


def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''

    pass


def c():
    return random.randint(0, 255)


def izracunaj_centre(slika, n=1, big_array=False, manual=True, t=5):
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
                    b = x
                    a = y
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

        colors = []

        while len(colors) < n:
            new = [c(), c(), c()]
            if all(dist(new, color) >= t for color in colors):
                colors.append(new)

        return colors

    return centre


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    k = 3
    iteracije = 3
    slika = cv.imread("./.utils/lenna.png")
    slika = cv.imread("./.utils/variety-of-peppers.png")
    # cv.cvtColor(slika, cv.COLOR_BGR2HSV)

    # slika = cv.resize(slika, (500, 350))

    segmentirana_slika = kmeans(slika, izracunaj_centre(slika, k, False, True, 1), k, iteracije)
    # segmentirana_slika = kmeans(slika, izracunaj_centre(slika, k, False, False, 1), k, iteracije)

    cv.setWindowTitle("slika", "Normalna slika")
    cv.imshow("slika", slika)
    cv.waitKey(0)

    cv.setWindowTitle("slika", "Slika obdealana s KMEANS algoritmom")
    cv.imshow("slika", segmentirana_slika)
    cv.waitKey(0)

    cv.destroyAllWindows()
    pass
