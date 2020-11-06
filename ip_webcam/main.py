from urllib import request
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_IMAGE():

    img = cv.imread('test.png', 0) # -1 Farbparameter
    img = img[33:412, 93:599]
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

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

    titel = ['Original', 'Binary']
    bilder = [img, thresh1]

    for k, i in enumerate(bilder):
        plt.subplot(1, 2, k+1)
        plt.imshow(i, 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titel[k])

    plt.show()

def get_VIDEO():

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', frame)
        cv.imshow('gray', gray)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    url = 'http://...'
    get_IMAGE()
    get_VIDEO()