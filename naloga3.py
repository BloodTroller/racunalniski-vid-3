import math
import random
import sys
import time
import warnings
from scipy.ndimage import gaussian_filter
import numba
from numba import jit
import cv2 as cv
import numpy as np


@jit(nopython=True)
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
                    nums[centre_index][3] += x2
                    nums[centre_index][4] += y2
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


def rng_c():
    return random.randint(0, 255)


def rng_xy(size):
    return random.randint(0, size - 1)


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
    elif big_array:
        colors = []
        distances = []
        output = []

        while len(colors) < n:
            new = [rng_c(), rng_c(), rng_c()]
            n1 = rng_xy(slika.shape[1])
            n2 = rng_xy(slika.shape[0])
            new2 = [n1, n2]
            new3 = new.copy()
            new3.append(n1)
            new3.append(n2)
            if all(dist(new, color) >= t for color in colors) and all(
                    dist(new2, distance) >= t for distance in distances):
                colors.append(new)
                distances.append(new2)
                output.append(new3)
        return output
    else:
        colors = []

        while len(colors) < n:
            new = [rng_c(), rng_c(), rng_c()]
            if all(dist(new, color) >= t for color in colors):
                colors.append(new)

        return colors

    return centre


def meanshift(slika, h, big_array=False, sigma=1.5, max_iteracije=10, m=0.02, dimeznija=3):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    output = slika.copy()
    height, width = slika.shape[:2]
    for x in range(height):
        for y in range(width):
            iteracija = 0
            konvergenca = False
            while not konvergenca and iteracija < max_iteracije:
                print("(", x, ",", y, ") Iteracija:", iteracija, "\n")
                tocka = output[x, y]
                razdalje = np.zeros((height, width))
                razdalje.fill(-1)
                # razdalje = izracunaj_razdaljo(tocka, vse_tocke)
                for x2 in range(height):
                    for y2 in range(width):
                        razdalje[x2, y2] = dist(output[x2, y2], tocka)
                # jedro(razdalje, h)
                utezi = (1 / (h * math.sqrt(2 * math.pi))) * np.exp(-1 / 2 * pow((razdalje / h), 2))
                # nova_tocka = sum(utezi * slika[x, y]) / sum(utezi)
                nova_tocka = np.sum(utezi[..., np.newaxis] * slika, axis=(0, 1)) / np.sum(utezi)
                konvergenca = dist(nova_tocka, tocka) < m
                output[x, y] = nova_tocka
                iteracija += 1
    return output


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    k = 5
    iteracije = 1
    slika = cv.imread("./.utils/zelenjava.jpg")
    h, w = slika.shape[:2]
    # slika = cv.imread("./.utils/variety-of-peppers.png")
    # cv.cvtColor(slika, cv.COLOR_BGR2HSV)

    # slika = cv.resize(slika, (500, 350))

    izbrane_barve = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    centri_za_papriko3 = [[47, 16, 197], [16, 185, 246], [30, 128, 235], [71, 137, 108], [255, 255, 255]]
    centri_za_papriko5 = [[47, 16, 197, 474, 212], [16, 185, 246, 214, 281], [30, 128, 235, 15, 922],
                          [71, 137, 108, 614, 777], [255, 255, 255, 64, 1198]]
    sslika = kmeans(slika, izbrane_barve, k, iteracije)
    # sslika = kmeans(slika, centri_za_papriko5, k, iteracije)
    # sslika = kmeans(slika, izracunaj_centre(slika, k, False, True, 1), k, iteracije)
    # sslika = kmeans(slika, izracunaj_centre(slika, k, False, False, 10), k, iteracije)

    #
    # slika = cv.resize(slika, (64, 64))
    # sslika = meanshift(slika, 1)
    #
    # slika = cv.resize(slika, (h, w))
    # sslika = cv.resize(sslika, (h, w))

    cv.setWindowTitle("slika", "Normalna slika")
    cv.imshow("slika", slika)
    cv.waitKey(0)

    cv.setWindowTitle("slika", "Slika obdealana z algoritmom")
    cv.imshow("slika", sslika)
    cv.waitKey(0)

    cv.destroyAllWindows()
    pass
