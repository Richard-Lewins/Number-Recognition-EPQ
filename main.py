from tkinter import *

import trainWindow
import testWindow


root = Tk()
root.title("Character Recognition Menu")
lblIntro = Label(root,text="This is character recognition by Richard Lewins:\n")
btnTrain = Button(root,text="Train The Model",width=20,height=5,command=trainWindow.openTrain)
btnTest = Button(root,text="Test Your Own Digits",width=20,height=5,command=testWindow.openTest)

lblIntro.pack()
btnTrain.pack(pady=5)
btnTest.pack(pady=5)



root.mainloop()
