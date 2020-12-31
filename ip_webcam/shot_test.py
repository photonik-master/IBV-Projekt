# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:32:54 2020

@author: simon
"""

import serial
import time
import schedule
import cv2

def main_func():
    global frame 
    global innitframe
    arduino = serial.Serial('com12', 9600)
    print('Openconnection to Arduino')
    print('')
    arduino_data = arduino.readline()
    print(arduino_data)
    decoded_values = str(arduino_data[0:len(arduino_data)].decode("utf-8"))
    
   # list_values = decoded_values.split('x')

    #for item in list_values:
   #     list_in_floats.append(float(item))

  #  print(f'Collected readings from Arduino: {list_in_floats}')
    #print(decoded_values)
    #print(type(decoded_values))
    
    
    
    if '1' in decoded_values:
        
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Display the resulting frame
        print(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(0) 
    if '2' in decoded_values:
        ret, innitframe = cap.read()
        print(innitframe)
        cv2.imshow('innitframe', innitframe)
        cv2.waitKey(0) 
    else:
        frame=None
           

        
        
        
        
    arduino_data = 0
    
   
   # list_in_floats.clear()
  #  list_values.clear()
    arduino.close()
    print('Connection closed')
    print('<----------------------------->')
    





# ----------------------------------------Main Code------------------------------------
# Declare variables to be used
#list_values = []
#list_in_floats = []

print('Program started')

cap = cv2.VideoCapture(0)


# Setting up the Arduino
schedule.every(2).seconds.do(main_func)

while True:
   schedule.run_pending()
   time.sleep(1)

cap.release()