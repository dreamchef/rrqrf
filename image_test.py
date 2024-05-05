# Author: Wren Taylor

import imageio as iio
import numpy as np
from matplotlib import pyplot
from skimage import color, feature, transform
from PIL import Image as im 

image = iio.imread('imageio:chelsea.png')
print(image)  # (300, 451, 3)
print(image[0, 0]) #this outputs the RGB values of the pixel at the given location 
pyplot.imshow(image)
pyplot.show()

img_gray = color.rgb2gray(image)
img_gray_normalized = img_gray / np.max(img_gray)  
pyplot.imshow(img_gray_normalized, cmap=pyplot.cm.gray)
pyplot.title('Grayscale Image')
pyplot.show()

#This is just seeing if the values are valid 
#color_test = np.full((10, 10, 3), 0.58914824)
#pyplot.imshow(color_test)
#pyplot.show()

A = np.array(image)
B = img_gray
print('BL:', B)
print('A:', A)
print(A.shape)

#pyplot.imshow(img_gray, cmap=pyplot.cm.gray)
#pyplot.show()


#This subroutine convert
def main(): 
  
    #array = B
      
    array = (img_gray_normalized * 255).astype(np.uint8)


    # check type of array 
    print(type(array)) 
      
    # our array will be of width  
    # 737280 pixels That means it  
    # will be a long dark line 
    print(array.shape) 
      
    # Reshape the array into a  
    # familiar resoluition 
    #array = np.reshape(array, (1024, 720)) 
      
    # show the shape of the array 
    print(array.shape) 
  
    # show the array 
    print('This is the array')
    print(array) 
      
    # creating image object of 
    # above array 
    data = im.fromarray(array) 
    print(data)

    pyplot.imshow(data, cmap=pyplot.cm.gray)
    pyplot.title('Reconverted to image')
    pyplot.show()
    # saving the final output  
    # as a PNG file 
    #data.save('gfg_dummy_pic.png') 

    
  
# driver code 
if __name__ == "__main__": 
    
  # function call 
  main() 