from tkinter import *
from tkinter import messagebox
import tensorflow as tf
import model

from PIL import Image, ImageDraw
import numpy as np
import cv2
import math
from scipy import ndimage

width = 25 #Width of lines for drawing

def openTest():
    global lastx, lasty

    if model.modelTrained[0] == False:
        messagebox.showwarning(title="Unable To Test", message="You need to train the AI model before you can test it!")
        return
        

    topTest = Toplevel()
    topTest.title("Test Your Own Digits")
    
    #Canvas
    drawingFrame = LabelFrame(topTest,text="Drawing",padx=10,pady=10)
    drawingFrame.pack(padx=5,pady=5)

    def activate_paint(e):
        global lastx, lasty
        cnv.bind('<B1-Motion>', paint)
        lastx, lasty = e.x, e.y

    def paint(e):
        global lastx, lasty
        x, y = e.x, e.y
        cnv.create_line((lastx, lasty, x, y), width=width)

        #  --- PIL
        draw.line((lastx, lasty, x, y), fill='black', width=width)
        lastx, lasty = x, y
    
    def clear():
        cnv.delete("all")
        draw.rectangle((0,0,420,420),fill='white')

    def predict():

        #PreProcessing of input image (Resize, greyscale and Invert)
        gray = np.asarray(image1)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(255-gray, (28, 28))

        #Remove empty spaces on the outside of image
        while np.sum(gray[0]) == 0:
            gray = gray[1:] #Remove top

        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1) #Remove left

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1] #Remove bottom

        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1) #Remove left

        #Resize inner box to 20x20
        rows,cols = gray.shape

        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))

        
            #For shitfting
        def getBestShift(img):
            cy,cx = ndimage.center_of_mass(img)

            rows,cols = img.shape
            shiftx = np.round(cols/2.0-cx).astype(int)
            shifty = np.round(rows/2.0-cy).astype(int)

            return shiftx,shifty
        
        def shift(img,sx,sy):
            rows,cols = img.shape
            M = np.float32([[1,0,sx],[0,1,sy]])
            shifted = cv2.warpAffine(img,M,(cols,rows))
            return shifted

        #Move image to centre
        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted

        #Add border whitespace around image to make 28x28
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        #Prediction:
        imPredict = tf.keras.utils.normalize(gray.reshape(1,28,28),axis=1)

        outPredict = model.tfModel.predict([imPredict])
        print('prediction -> ',np.argmax(outPredict[0]))
        messagebox.showinfo("Prediction Complete","Number predicted: " + str(np.argmax(outPredict[0])),parent=topTest)




    cnv = Canvas(drawingFrame, width=420, height=420, bg='white')
    cnv.bind('<1>', activate_paint)
    cnv.pack(expand=YES,fill=BOTH)

    lastx, lasty = None, None

    # --- PIL
    image1 = Image.new('RGB', (420, 420), 'white')
    draw = ImageDraw.Draw(image1)

    cnv.pack(expand=YES, fill=BOTH)

    #Bottom Buttons

    buttonsFrame = Frame(topTest,padx=5,pady=5)
    buttonsFrame.pack()
    
    btnPredict = Button(buttonsFrame,text="Predict",command=predict)
    btnPredict.grid(column=0,row=0,padx=5)

    btnClear = Button(buttonsFrame,text="Clear",command=clear)
    btnClear.grid(column=1,row=0,padx=5)




