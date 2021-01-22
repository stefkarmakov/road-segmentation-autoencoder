import cv2
import os
import matplotlib
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import array_to_img
from natsort import natsorted

# create prediction set of images
vid = 'video_path.mp4' # insert video path
vid_folder = os.getcwd() + '\\' + vid.rpartition('.')[0] # create new folder path for the video frames
if not os.path.exists(vid_folder): # create the new folder
    os.makedirs(vid_folder)

vidcap = cv2.VideoCapture(vid)
success,img = vidcap.read()
count = 0
vid_images = []
while success:
    img_resized = cv2.resize(img, (256,256)) # reseze all images
    cv2.imwrite(os.path.join(vid_folder , "frame{}.png".format(count)), img_resized) # save frame as image    
    vid_images.append(img_resized) # save images to list
    success,img = vidcap.read()
    count += 1
  
vidcap.release()
cv2.destroyAllWindows()

tensor_shape = (-1,) + np.shape(vid_images[0])
vid_images = np.reshape(vid_images, (tensor_shape))/255. # turn frames list to tensor and normalize

#%% 
# load the trained model
json_file = open(r'path_to_model.json','r') # insert model JSON file path
model_json = json_file.read() 
json_file.close()
autoencoder = model_from_json(model_json)
autoencoder.load_weights(r"path_to_model.h5") # insert model H5 file path
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

#%%
# Apply segmentation to the recored video
prediction = autoencoder.predict(vid_images) # segment each frame
array_to_img(prediction[600]) # visualize a frame for validation

vid_folder_seg = os.getcwd() + '\\' + vid.rpartition('.')[0] + '_seg' # new segmented images folder
if not os.path.exists(vid_folder_seg):
    os.makedirs(vid_folder_seg)

count = 0
for each_image in range(len(prediction)):
    cv2.imwrite(os.path.join(vid_folder_seg , "frame{}_seg.png".format(count)), 
                prediction[each_image]*10e4) # save segmented images and scale values to be 0-255 for rgb
    count += 1

#%%    
# Create segmented video
image_folder = vid_folder_seg
video_name = vid.rpartition('.')[0] + '_seg3.avi'
# read sorted frames
frame_seg_images = [img for img in natsorted(os.listdir(image_folder)) if img.endswith(".png")] 
# get the shape of one frame for the video dims
height, width, layers = (cv2.imread(os.path.join(image_folder, frame_seg_images[0]))).shape 
video = cv2.VideoWriter(video_name, 0, 25, (width,height)) # create video with fps = 25

for each_frame in frame_seg_images:
    video.write(cv2.imread(os.path.join(image_folder, each_frame)))

cv2.destroyAllWindows()
video.release()







