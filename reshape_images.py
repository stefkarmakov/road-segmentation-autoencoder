import cv2
import os

original_folder = 'D:\\General_Projects\\Autoencoders\\ADE20K_2016_07_26\\images\\training\\h\\highway' # insert folder containing highway/street data
save_folder = 'D:\\General_Projects\\Autoencoders\\autoencoder_highway_color_data' # insert folder where cleaned and reshaped images will be stored
new_img_size = 256
for filename in os.listdir(original_folder):
    # change key markers for removing unneccesary images based on project
    if filename[filename.index('.')+1:] == 'txt' or 'part' in filename: 
        continue
    img = cv2.imread(os.path.join(original_folder,filename))
    img = cv2.resize(img, (new_img_size,new_img_size))
    cv2.imwrite(os.path.join(save_folder,filename), img)
    