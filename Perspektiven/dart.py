#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


def corrected(img1, img2): #perspektive von bild 2 wird auf bild 1 angepasst. (korrigiert)
    MIN_MATCHES = 50
    
    orb = cv2.ORB_create(nfeatures=500) #500features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: #wie gut liegen die features aneinander
            good_matches.append(m)
        
    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))
        return corrected_img
    return False


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

# define path from image - r means raw string, else unicode error
paths = [r"C:\Users\tobil\Downloads\dart_leer.png", r"C:\Users\tobil\Downloads\dart_drin.png"]
path2 = [r"C:\Users\tobil\Downloads\drin2.jpg", r"C:\Users\tobil\Downloads\draus2.jpg"]

# load image from paths
img1, img2 = cv2.imread(path2[0]), cv2.imread(path2[1])

# color correction - matplotlib and cv2 use different channels
img1, img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# show pics
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()


# In[4]:


new_img = corrected(img2, img1) # bei beiden bilder wird die Perspektive angepasst

if new_img is not False:
    plt.imshow(new_img) #das 
    plt.show()


# In[5]:


diff = cv2.absdiff(img1, new_img) #nimmt das ausgangsbild und vergleicht das mit dem neuen

plt.imshow(diff)
plt.show()

mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

plt.imshow(mask)
plt.show()


# In[22]:


# Convert to grayscale. 
circle_found = img2.copy()

gray = cv2.cvtColor(circle_found, cv2.COLOR_BGR2GRAY) 
  
# Blur using 3 * 3 kernel. 
gray_blurred = cv2.medianBlur(gray, 25) 
  
# Apply Hough transform on the blurred image. 
detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 115, minRadius = 800, maxRadius = 1200) 
  
# Draw circles that are detected. 
if detected_circles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(circle_found, (a, b), r, (0, 255, 0), 2) 

plt.figure(figsize = (20,20))
plt.imshow(gray_blurred)
plt.show()
        
plt.figure(figsize = (20,20))
plt.imshow(circle_found)
plt.show()

