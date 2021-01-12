import cv2 as cv
import matplotlib as plt
import numpy as np
import test
import game


# def draw_board(self):
#
#     shape = self.img.shape
#     # height, width = 400, 400
#     blank = np.zeros((shape[0], shape[1]), dtype=np.uint8)
#     # blank = np.zeros(shape=[height, width, 3], dtype=np.uint8)
#
#     for i in self.rad:
#         cv.circle(blank, self.cir_center, i, (255, 255, 255), 1, cv.LINE_8, 0)
#
#     a = np.arange(0 + self.angle_offset, 360 + self.angle_offset, 18)
#
#     for angle in a:
#         x2 = int(self.cir_center[0] + self.line_length * np.cos(np.radians(angle)))
#         y2 = int(self.cir_center[1] + self.line_length * np.sin(np.radians(angle)))
#         cv.line(blank, self.cir_center, (x2, y2), (255, 255, 255), 1, cv.LINE_8, 0)
#
#     cv.imshow('Board', blank)
#     cv.waitKey(3000)
#     cv.destroyAllWindows()


def order_points(pts):
    pts = np.array(pts)
    pts.shape = (4, 2)

    rect = np.zeros((4, 2), dtype="float32")
    # print(rect)

    s = np.sum(pts, axis=1)
    # print(s)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # print(rect)

    d = np.diff(pts, axis=1)
    # print(d)

    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    print(rect)

    return rect


def is_inside_circle(center, radius, point):
    if ((point[0] - center[0]) * (point[0] - center[0]) +
            (point[1] - center[1]) * (point[1] - center[1]) <= radius * radius):
        return True
    else:
        return False


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # # compute the width of the new image, which will be the
    # # maximum distance between bottom-right and bottom-left
    # # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # # compute the height of the new image, which will be the
    # # maximum distance between the top-right and bottom-right
    # # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # # now that we have the dimensions of the new image, construct
    # # the set of destination points to obtain a "birds eye view",
    # # (i.e. top-down view) of the image, again specifying points
    # # in the top-left, top-right, bottom-right, and bottom-left
    # # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def draw_board_new(image):
    # blank = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    # height, width = blank.shape
    # cen = (int(round(width / 2)), int(round(height / 2)))

    cv.ellipse(image, (840, 1160), (25, 30), 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)

    cv.drawMarker(image, (865, 1160), color=(255, 255, 255), markerType=cv.MARKER_CROSS, thickness=3)
    print(is_inside_ellipse((865, 1160), (840, 1160), (25, 30)))

    cv.ellipse(image, (840, 1160), (55, 70), 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.ellipse(image, (810, 1155), (320, 430), 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.ellipse(image, (810, 1155), (360, 470), 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.ellipse(image, (770, 1155), (535, 710), 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)
    cv.ellipse(image, (762, 1152), (570, 758), 0, 0, 360, (255, 255, 255), 1, cv.LINE_8, 0)

    length = 800
    cen = (840, 1160)
    # offset = 9
    # a = np.arange(0 + offset, 360 + offset, 18)
    a = [10, 33, 52, 69, 83, 96, 111, 126, 145, 167, 191, 213, 232, 248, 262, 276, 290, 306, 325, 346]

    for angle in a:
        x2 = int(cen[0] + length * np.cos(np.radians(angle)))
        y2 = int(cen[1] + length * np.sin(np.radians(angle)))
        cv.line(image, cen, (x2, y2), (255, 255, 255), 1, cv.LINE_8, 0)


    return image


def is_inside_ellipse(point, center, rad):
    a = ((point[0] - center[0]) ** 2) / (rad[0]**2)
    b = ((point[1] - center[1]) ** 2) / (rad[1]**2)

    if (a + b) < 1:
        return True
    elif (a + b) == 1:
        return '!!!'
    else:
        return False


if __name__ == '__main__':
    img = test.get_IMAGE('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_22.png', 0)
    # test.view_image(img, 'Bild')
    # cv.waitKey(1)
    # cv.destroyAllWindows()

    bo = draw_board_new(img)
    test.view_image(bo, 'Board')
    cv.waitKey(1)
    cv.destroyAllWindows()


    # img = test.get_IMAGE('/Users/alex/Workspace/git_repos/IBV-Projekt/darts/bilder/arrow_211.png', 0)
    # test.view_image(img, 'Bild')
    # cv.waitKey(1)
    # cv.destroyAllWindows()
    #
    # pts = [(144, 284), (1404, 484+50), (1404, 1856-50), (184, 2032)]
    # # order_points(pts)
    # trans = four_point_transform(img, pts)
    # test.view_image(trans, 'Trans')
    # cv.waitKey(1)
    # cv.destroyAllWindows()

    # pts = []
    #
    # draw_board()
    #
    # center = (0, 0)
    # radius = 200
    # point = (5, 5)
    #
    #
    # print(is_inside_circle(center, radius, point))
    #
