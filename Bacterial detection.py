#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code is based on examples from https://realpython.com/pysimplegui-python/

import PySimpleGUI as sg
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')

# Create the setup
def main():
    sg.theme("LightBlue6")

    # Define the window layout
    layout = [
        [sg.Image(filename="", key="-IMAGE-")],
        [
            sg.Button("LOG", size=(10, 1)),
            sg.Slider(
                (-10, 10),
                0,
                0.1,
                orientation="h",
                size=(40, 10),
                key="-LOG SLIDER-",
            ),
            sg.Button("GAMMA", size=(10, 1)),
            sg.Slider(
                (0, 25),
                1,
                0.1,
                orientation="h",
                size=(40, 10),
                key="-GAMMA SLIDER-",
            ),
            
        ],
        [
            sg.Button("AVERAGE", size=(10, 1)),
            sg.Slider(
                (1, 21),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-BLUR SLIDER-",
            ),
            sg.Button("MEDIAN", size=(10, 1)),
            sg.Slider(
                (1, 21),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-MEDIAN SLIDER-",
            ),
            
        ],
        [
            sg.Button("HSV_THS", size=(10, 1)),
            sg.Text('H low'),
            sg.Slider(
                (0, 179),
                90,
                1,
                orientation="h",
                size=(15, 10),
                key="-HSV SLIDER H LOW-",
            ),
            sg.Text('H high'),
            sg.Slider(
                (0, 179),
                179,
                1,
                orientation="h",
                size=(15, 10),
                key="-HSV SLIDER H HIGH-",
            ),
            sg.Text('S Low'),
            sg.Slider(
                (0, 255),
                125,
                1,
                orientation="h",
                size=(18, 10),
                key="-HSV SLIDER S LOW-",
            ),
            sg.Text('S High'),
            sg.Slider(
                (0, 255),
                255,
                1,
                orientation="h",
                size=(18, 10),
                key="-HSV SLIDER S HIGH-",
            ),
            sg.Text('V Low'),
            sg.Slider(
                (0, 255),
                125,
                1,
                orientation="h",
                size=(18, 10),
                key="-HSV SLIDER V LOW-",
            ),
            sg.Text('V High'),
            sg.Slider(
                (0, 255),
                255,
                1,
                orientation="h",
                size=(18, 10),
                key="-HSV SLIDER V HIGH-",
            ),
        ],
        [
            sg.Button("ERODE", size=(10, 1)),
            sg.Slider(
                (1, 15),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-ERODE SLIDER-",
            ),
            sg.Button("DILATE", size=(10, 1)),
            sg.Slider(
                (1, 15),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-DILATE SLIDER-",
            ),
            
        ],
        [sg.Button("Reset changes", size=(12, 1)),sg.Button("Histogram", size=(10, 1)),sg.Button("Outline", size=(10, 1)),sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window
    window = sg.Window("Bacteria detection", layout, location=(800, 400))# 800,400 - default
    
    img = cv2.imread('test3.jpg')# Change the file here by typing the filename
    img = cv2.resize(img, (800, 600))
    
    #(M, N, channels) = img.shape # can be used to check whether the image is read or not
    #print(M, N)
    
    image = img.copy()
    img_tmp = img.copy()
    
    frame = np.concatenate((img_tmp, image), axis=1)
    
    # Create event loop
    while True:
        event, values = window.read(timeout=200)
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            print("exit")
            break
       
        elif event == "Reset changes":
            img_tmp = img.copy()
            frame = np.concatenate((img_tmp, image), axis=1)
            print('ResetRGB')
        
        # Image restoration filters
        elif event == "AVERAGE":
            b_val = int(values["-BLUR SLIDER-"])
            if (b_val % 2) == 0:
                b_val = b_val+1
            img_tmp = cv2.blur(img_tmp, (b_val, b_val), )
            frame = np.concatenate((img_tmp, image), axis=1)
            print('average')            
       
        elif event == "MEDIAN":
            m_val = int(values["-MEDIAN SLIDER-"])
            if (m_val % 2) == 0:
                m_val = m_val+1
            img_tmp = cv2.medianBlur(img_tmp, m_val)
            frame = np.concatenate((img_tmp, image), axis=1)
            print('median') 
        
        #Thresholding
        elif event == "HSV_THS":
            img_hsv = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2HSV)
            lower = np.array([int(values["-HSV SLIDER H LOW-"]),int(values["-HSV SLIDER S LOW-"]),int(values["-HSV SLIDER V LOW-"])])
            upper = np.array([int(values["-HSV SLIDER H HIGH-"]),int(values["-HSV SLIDER S HIGH-"]),int(values["-HSV SLIDER V HIGH-"])])
            mask = cv2.inRange(img_hsv,lower,upper)
            
            # You can leave the thresholded image in black and white
            #img_tmp = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            
            # Initially, the thresholded mask is applied to image allowing the preservation of bacteria colour
            masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask = mask)
            img_tmp = cv2.cvtColor(masked_img,cv2.COLOR_HSV2BGR)
            frame = np.concatenate((img_tmp, image), axis=1)
            print('HSV_THS')   
        
        # Image enhancement functions 
        elif event == "ERODE":
            e_val = int(values["-ERODE SLIDER-"])
            kernel = np.ones((e_val,e_val),np.uint8)
            img_tmp = cv2.erode(img_tmp,kernel,iterations = 1)
            frame = np.concatenate((img_tmp, image), axis=1)
            print('erode')
        
        elif event == "DILATE":
            d_val = int(values["-DILATE SLIDER-"])
            kernel = np.ones((d_val,d_val),np.uint8)
            img_tmp = cv2.dilate(img_tmp,kernel,iterations = 1)
            frame = np.concatenate((img_tmp, image), axis=1)
            print('dilate')
        
        # Image transformations functions 
        elif event == "LOG":
            const = values["-LOG SLIDER-"]
            # For a gray-scale logarithmic transformation 
            #img_tmp = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2GRAY)
            img_tmp = img_tmp.astype('float')
            img_tmp = ((const * (np.log10(img_tmp + 1)))*255).astype("uint8")
            #img_tmp = cv2.cvtColor(img_tmp,cv2.COLOR_GRAY2BGR)
            frame = np.concatenate((img_tmp, image), axis=1)
            print('log')
        
        elif event == "GAMMA":
            c = 1.0
            gamma = values["-GAMMA SLIDER-"]
            print(gamma)
            img_tmp = (c*((img_tmp/ 255.0)**gamma)*255).astype("uint8")
            frame = np.concatenate((img_tmp, image), axis=1)
            print('gamma')
        
        #Draw and display detected bacteria outline
        elif event == "Outline":
            contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_tmp = cv2.drawContours(img_tmp, contours, -1, (180,130,70),2)
            frame = np.concatenate((img_tmp, image), axis=1)
            print('outline') 
        
        #Display statistics about bacteria area (pixel count) as a histogram (x axis - area, y axis bacteria count)
        elif event == "Histogram":
            contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas=np.empty([0, 0])
            for contour in range(0,len(contours)):
                areas = np.append(areas,round(cv2.contourArea(contours[contour]),2))
            
            # If one wants to check the existed areas to choose the best x limit range, then this line can be uncommented:
            #print('Areas:',areas)
            
            hist, bins = np.histogram(areas,range(0,len(areas)))
            plt.hist(hist, bins, histtype = 'stepfilled')
        
            plt.xlabel("area")
            
            # The limit can be changed by user
            plt.xlim(-10,100)
            
            plt.ylabel("number of cells")
            
            plt.title("Histogram")
            plt.show()
            print('hist') 
            
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()

