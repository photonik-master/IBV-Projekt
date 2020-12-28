import cv2 as cv
import test_einlesen

def computeDifference(grey1, grey2):

    # blur
    blur = 5
    grey2 = cv.medianBlur(grey2, blur)
    grey1 = cv.medianBlur(grey1, blur)
    # normalize
    grey1 = cv.equalizeHist(grey1)
    grey2 = cv.equalizeHist(grey2)
    clahe = cv.createCLAHE(20, (10, 10))
    # clahe
    grey1 = clahe.apply(grey1)
    grey2 = clahe.apply(grey2)
    # #diff
    diff = cv.subtract(grey2, grey1) + cv.subtract(grey1, grey2)
    ret2, dif_thred = cv.threshold(diff, 75, 255, cv.THRESH_BINARY)

    return dif_thred, grey1, grey2, diff

if __name__ == '__main__':

    bild1 = test_einlesen.get_IMAGE('bilder/20201208_141922.jpg', 0)
    bild2 = test_einlesen.get_IMAGE('bilder/20201208_141923.jpg', 0)

    dif_thred, grey1, grey2, diff = computeDifference(bild1, bild2)

    title = ['grey1', 'grey2', 'dif_thred', 'diff']
    bilder = [dif_thred, grey1, grey2, diff]

    for k, i in enumerate(bilder):
        cv.namedWindow(title[k], cv.WINDOW_NORMAL)
        cv.moveWindow(title[k], 20, 20)
        cv.resizeWindow(title[k], 600, 600)
        cv.imshow(title[k], i)

    cv.waitKey(0)
    cv.destroyAllWindows()

