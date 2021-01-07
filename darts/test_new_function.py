import cv2 as cv
import matplotlib as plt
import numpy as np

# def isInside(circle_x, circle_y, rad, x, y):
#     # Compare radius of circle
#     # with distance of its center
#     # from given point
#     if ((x - circle_x) * (x - circle_x) +
#             (y - circle_y) * (y - circle_y) <= rad * rad):
#         return True;
#     else:
#         return False;

def zeichne():

    r = 100
    blank = np.zeros((5 * r, 5 * r), dtype=np.uint8)
    # blank = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    height, width = blank.shape
    center = (int(round(width/2)), int(round(height/2)))

    cv.circle(blank, center, 10, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.circle(blank, center, 15, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.circle(blank, center, 100, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.circle(blank, center, 110, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.circle(blank, center, 200, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.circle(blank, center, 210, (255, 255, 255), 1, cv.LINE_8, 0)

    length = 210
    offset = 9
    p1 = center
    a = np.arange(0 + offset, 360 + offset, 18)

    for angle in a:
        x2 = int(p1[0] + length * np.cos(np.radians(angle)))
        y2 = int(p1[1] + length * np.sin(np.radians(angle)))
        cv.line(blank, center, (x2, y2), (255, 255, 255), 1, cv.LINE_8, 0)

    # Get the contours
    # contours, hierarchy = cv.findContours(blank, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #
    # img_contour = blank.copy()
    # for i in range(len(contours)):
    #     print(i)
    #     img_contour = cv.drawContours(img_contour, contours[i], -1, (0, 255, 0), 5)

    # # Calculate the distances to the contour
    # raw_dist = np.empty(src.shape, dtype=np.float32)
    # for i in range(src.shape[0]):
    #     for j in range(src.shape[1]):
    #         raw_dist[i, j] = cv.pointPolygonTest(contours[0], (j, i), True)
    # minVal, maxVal, _, maxDistPt = cv.minMaxLoc(raw_dist)
    # minVal = abs(minVal)
    # maxVal = abs(maxVal)
    # # Depicting the  distances graphically
    # drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    # for i in range(src.shape[0]):
    #     for j in range(src.shape[1]):
    #         if raw_dist[i, j] < 0:
    #             drawing[i, j, 0] = 255 - abs(raw_dist[i, j]) * 255 / minVal
    #         elif raw_dist[i, j] > 0:
    #             drawing[i, j, 2] = 255 - raw_dist[i, j] * 255 / maxVal
    #         else:
    #             drawing[i, j, 0] = 255
    #             drawing[i, j, 1] = 255
    #             drawing[i, j, 2] = 255
    # cv.circle(drawing, maxDistPt, int(maxVal), (255, 255, 255), 1, cv.LINE_8, 0)
    cv.imshow('Source', blank)
    # cv.imshow('Distance and inscribed circle', drawing)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':

    zeichne()

    cv.waitKey(0)
    cv.destroyAllWindows()
