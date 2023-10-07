"""
The main script of the Dicinger Pro Max.
"""
from scripts.application import Application

if __name__ == '__main__':
    app = Application()

    # Make the GUI size enough and in the center of the screen.
    width = 1800
    height = 1200
    screenwidth = app.winfo_screenwidth()
    screenheight = app.winfo_screenheight()
    screenX = (screenwidth-width)/2
    screenY = (screenheight-height)*1/3
    app.geometry("%dx%d+%d+%d" % (width, height, screenX, screenY))

    # create the application mainloop
    app.mainloop()