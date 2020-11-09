from urllib import request
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_IMAGE():

    img = cv.imread('test.png', -1) # -1-Farbparameter
    img = img[33:412, 93:599]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret0, thresh0 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    #ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #gauss = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)

    kernel =np.ones((2, 2), np.uint8)
    erosion = cv.erode(thresh0, kernel, iterations=1)
    dilation = cv.dilate(thresh0, kernel, iterations=1)

    erg = cv.dilate(erosion, kernel, iterations=1)

    print(kernel)

    #cv2.line(img, (0, 0), (150, 150), (0, 255, 0), 10) #Linie
    #cv2.rectangle(img, (15, 25), (200, 150), (0, 250, 0), 5)   #Rechteck
    #cv2.circle(img, (100, 63), 55, (0, 255, 0), -1)    #Kreis
    #pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)  #P_gon
    #cv2.polylines(img, [pts], True, (0, 255, 0), 1)    # True Parameter (start-stop) verbinden
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, 'TEST!', (0, 130), font, 1, (0, 255, 0))

    #cv.imshow('Bild', img)
    #cv.imshow('binar', thresh1)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #cv.imwrite('neu.png', img)

    titel = ['Original', 'Gray', 'Binary', 'Erosion', 'Dilation', 'ERG']
    bilder = [img, gray, thresh0, erosion, dilation, erg]

    for k, i in enumerate(bilder):
        plt.subplot(2, 3, k+1)
        plt.imshow(i, 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titel[k])

    plt.show()

def get_VIDEO():

    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        # hue sat value
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        lower_col = np.array([30, 100, 100])
        upper_col = np.array([330, 255, 255])

        mask = cv.inRange(hsv, lower_col, upper_col)
        res = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)

        #cv.imshow('gray', gray)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    url = 'http://...'
    #get_IMAGE()
    get_VIDEO()