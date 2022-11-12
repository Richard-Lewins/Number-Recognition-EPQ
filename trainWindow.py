from tkinter import *
from tkinter import scrolledtext,ttk
import model
import testWindow

import math
import tensorflow as tf
import time
import threading


def openTrain():
    global topTrain
    global settingsFrame
    global entLearningRate,spbBatchSize,spbEpochs
    global setBatchSize,setEpochs
    global txtOutput
    global pb
    global btnTest

    topTrain = Toplevel()
    topTrain.title("Train the model")
    settingsFrame = LabelFrame(topTrain,text="Settings",padx=50,pady=10)
    settingsFrame.pack(padx=5,pady=5)

    #Adjust Learning Rate
    lblLearningRate = Label(settingsFrame,text="Learning Rate:")
    lblLearningRate.grid(column=0,row=0,sticky='w',padx=(0,10))

    entLearningRate = Entry(settingsFrame)
    entLearningRate.grid(column=1,row=0,sticky='w')
    entLearningRate.insert(0,"0.001")

    #Adjust Batch Size
    lblBatchSize = Label(settingsFrame,text="Batch Size (1-60000):")
    lblBatchSize.grid(column=0,row=1,sticky='w',padx=(0,10))

    setBatchSize = StringVar(settingsFrame)
    spbBatchSize = Spinbox(settingsFrame,from_=1,to=60000,textvariable = setBatchSize)
    spbBatchSize.grid(column=1,row=1,sticky='w')
    setBatchSize.set("32")

    #Adjust Number of Epochs
    lblEpochs = Label(settingsFrame,text="Epochs: ")
    lblEpochs.grid(column=0,row=3)

    setEpochs = StringVar(settingsFrame)
    spbEpochs = Spinbox(settingsFrame,from_=1 ,to=10000, textvariable=setEpochs)
    spbEpochs.grid(column=1,row=3,sticky='w')
    setEpochs.set('3')


    #Add output field
    outputFrame = LabelFrame(topTrain,text="Output",padx=10,pady=10)
    outputFrame.pack(padx=5,pady=5)
    txtOutput = scrolledtext.ScrolledText(outputFrame,width=40)
    txtOutput.pack()

    btnTrain = Button(topTrain,text="Train Model",command=btnTrainClick)
    btnTrain.pack(pady=5)
    
    #Progress bar for when training model
    pb = ttk.Progressbar(topTrain,orient='horizontal',mode='indeterminate',length=280)

    #Button to open test window after training is complete
    btnTest = Button(topTrain,text="Test your own digits",command=testWindow.openTest)

def btnTrainClick():

#Input Validation

    if model.bTraining == True: #If the model is currently being trained, do not continue
        txtOutput.insert(END,"Please be patient.\n")
        return

    try:
        iBatchSize = math.floor(float(spbBatchSize.get()))
    except ValueError:
        iBatchSize = 32

    try:
        iNumEpochs = math.floor(float(spbEpochs.get()))
    except ValueError:
        iNumEpochs = 3

    try: 
        rLearningRate = float(entLearningRate.get())
    except ValueError:
        rLearningRate = 0.001

    if rLearningRate <= 0:
        rLearningRate = 0.001


    if iBatchSize < 1:
        iBatchSize = 1

    if iBatchSize >60000:
        iBatchSize = 60000

    if iNumEpochs < 1:
        iNumEpochs = 3

    #Set the entrys/spinboxes to the values in cases of any changes during validation
    entLearningRate.delete(0,END)
    entLearningRate.insert(0,str(rLearningRate))
    setBatchSize.set(str(iBatchSize))
    setEpochs.set(str(iNumEpochs))

    #Outputing Information
    txtOutput.insert(END,"Learning Rate: " + str(rLearningRate)+'\n')
    txtOutput.insert(END,"Batch Size: " + str(iBatchSize)+'\n')
    txtOutput.insert(END,"# of Epochs: " + str(iNumEpochs)+ '\n')
    txtOutput.insert(END,"Training the model...\n\n")
    txtOutput.see(END)

    topTrain.update()

    #trainModel(rLearningRate,iBatchSize)
    t1 = threading.Thread(target=trainModel,args=(rLearningRate,iBatchSize,iNumEpochs))
    t1.start()

    #start progress bar
    pb.pack()
    pb.start()

#For model training thread
def trainModel(learningRate,batchSize,numEpochs):

    model.bTraining = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)

    #compiling model
    model.tfModel.compile(optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    #compiling model
    print("Training:")
    startTime = time.time()
    model.tfModel.fit(model.x_train,model.y_train,epochs=numEpochs,batch_size=batchSize)
    endTime = time.time()

    #Evaluating model
    timeTaken = endTime - startTime
    print("\nEvaluation")
    evalLoss,evalAcc = model.tfModel.evaluate(model.x_test,model.y_test)

    
    #Print Information about latest training round (numEpochs - 1)
    txtOutput.insert(END,"\nTraining Completed!\n")
    txtOutput.insert(END,"Accuracy: " + str(round(evalAcc*100,2)) + '%\n')
    txtOutput.insert(END,"Loss: " + str(round(evalLoss,4)) + '\n')
    txtOutput.insert(END,"Time Taken: " + str(round(timeTaken,5)) + " seconds\n\n")
    pb.pack_forget()

    if model.modelTrained[0] ==False:
        model.modelTrained[0] = True
        btnTest.pack(pady=5)

    model.bTraining = False




    







