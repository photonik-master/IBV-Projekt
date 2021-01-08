import cv2 as cv
import matplotlib as plt
import numpy as np


def is_inside_circle(center, radius, point):
    if ((point[0] - center[0]) * (point[0] - center[0]) +
            (point[1] - center[1]) * (point[1] - center[1]) <= radius * radius):
        return True
    else:
        return False


def draw_board():

    r = 100
    blank = np.zeros((5 * r, 5 * r), dtype=np.uint8)
    # blank = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    height, width = blank.shape
    cen = (int(round(width / 2)), int(round(height / 2)))
    rad = [10, 15, 100, 110, 200, 210]

    for i in rad:
        cv.circle(blank, cen, i, (255, 255, 255), 1, cv.LINE_8, 0)

    length = 210
    offset = 9
    a = np.arange(0 + offset, 360 + offset, 18)

    for angle in a:
        x2 = int(cen[0] + length * np.cos(np.radians(angle)))
        y2 = int(cen[1] + length * np.sin(np.radians(angle)))
        cv.line(blank, cen, (x2, y2), (255, 255, 255), 1, cv.LINE_8, 0)

    cv.imshow('Board', blank)
    cv.waitKey(5000)
    cv.destroyAllWindows()


if __name__ == '__main__':

    draw_board()

    center = (0, 0)
    radius = 200
    point = (5, 5)


    print(is_inside_circle(center, radius, point))

    cv.waitKey(1)
    cv.destroyAllWindows()
