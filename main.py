import numpy as np
import scipy
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import skimage
import warnings
from skimage import data
from IPython.display import Image

warnings.filterwarnings("ignore")
photo_data = imageio.imread('data/sd-3layers.jpg')
type(photo_data)

plt.imshow(photo_data)
plt.show()
#print(photo_data)

#print(photo_data.shape)

print("shape: ", photo_data.shape)
print("size: ", photo_data.size)
print("min: ", photo_data.min())
print("max: ", photo_data.max())
print("mean: ", photo_data.mean())


print("150, 250: ",photo_data[150, 250])
print("150,250,1: ",photo_data[150, 250, 1])

print("1,1: ", photo_data[1,1])

photo_data[150, 250] = 0
print("changed 150,250 to 0: ", photo_data[150,250])


#plt.figure(figsize=(15, 15))
#plt.imshow(photo_data)
#plt.show()

'''
photo_data[0:150, : ,1] = 255
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
plt.show()

photo_data[0:200, :] = 00
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
'''

#Pick all pixels with low values
print("shape of photo_data: ", photo_data.shape)
low_value_filter = photo_data < 100
print("shape of low_value_filter: ", low_value_filter.shape)

#Filtering out low values
'''
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
photo_data[low_value_filter] = 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
'''

#Row and column operations
#We can design complex patterns by making columns a function
# of rows or vice-versa. Here we try a linear relationship between rows and columns.
rows_range = np.arange(len(photo_data))
print(rows_range)
cols_range = rows_range
print(cols_range)
print(type(cols_range))

#setting selected rows and columns to 255
'''
photo_data[rows_range, cols_range] = 255
print(photo_data)
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
'''
#We see a diagonal white line that is a result of our operation.

#Masking Images
#Now let us try to mask the image in shape of a circular disc.

total_rows, total_cols, total_layers = photo_data.shape
print("photo_data shape :", photo_data.shape)
X, Y = np.ogrid[:total_rows, :total_cols]
print("X :", X.shape, "Y :", Y.shape)
Image("data/myimg.jpg")

center_row, center_col = total_rows/2, total_cols/2
print("center_row : ", center_row, "center_col : ", center_col)
print(X - center_row)
print(Y - center_col)

dist_from_center = (X - center_row)**2 + (Y - center_col)**2
print(dist_from_center)
radius = (total_rows/2)**2
print("Radius :", radius)
circular_mask = (dist_from_center > radius)
print("circular mask : ", circular_mask)
print("circular mask[1500:1700, 2000:2200", circular_mask[0:200, 10:12])

#Creating a circular mask which makes all pixels greater than the radius as 0(black)
'''
photo_data = imageio.imread('data/sd-3layers.jpg')
photo_data[circular_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
'''

#Further Masking
#We can further improve the mask, for example just get upper half disc.

#Creating a half mask which makes the upper semicircle only visible rest set to 255(white)
'''
half_upper = X < center_row
half_upper_mask = np.logical_and(half_upper, circular_mask)
photo_data[half_upper_mask] = 255
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
'''

#detecting highly red pixels

'''
red_mask = photo_data[:, :, 0] < 150
photo_data[red_mask] = 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
'''

#detecting highly green pixels
'''

green_mask = photo_data[:, :, 1] < 150
photo_data[green_mask] = 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
'''

#detecting highly blue pixels
'''
blue_mask = photo_data[:, :, 2] < 150
photo_data[blue_mask] = 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
'''

#Composite mask that takes thresholds on all three layers: RED, GREEN, BLUE
red_mask = photo_data[:,:,0] < 150
green_mask = photo_data[:,:,1] > 100
blue_mask = photo_data[:,:,2] < 100
final_mask = np.logical_and(red_mask, green_mask, blue_mask)
photo_data[final_mask] = 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data);
plt.show()
