from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def fig2data ( fig ):
    fig.canvas.draw ( )
 
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def PlotToImage(x_arr, y_arr):
    figure = plt.figure()
    plt.plot (x_arr, y_arr)
    return np.array(fig2img (figure))