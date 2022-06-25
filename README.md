# Road Segmentation Autoencoder

The following scripts use autoencoding to segment images from road and highway settings. The scripts give a flow and the Autoencoder architecture (Convolutional autoencoder), that you can use to train your own model with your data. The data that I used for training the model is the street and highway images from the MIT Scene Parsing Benchmark. 

1. Run the `reshape_images.py` script. It takes in a folder with images and saves only the orignal images and their segmented representations. Change `original_folder` and `save_folder` with the folder containing the original images, and the path to which the reshaped images should be saved. All images need to be the same size before being fed in the Autoencoder. Before saving, the script resizes them to 256x256 images, if the original ones are different sizes. The image sizes don't need to be 256x256, that's what I've used for the data I chose. For a different image size, just change the `new_img_size` param to the desired pixels size. 

2. Run the `segmentation_autoencoder.py` script. It trains a model on the saved images from `reshape_images.py`. For the script to recognize the images and their segmented vesions, the images need to be with `.jpg` extension and the segmented images need to end with `..._seg.png`. Ex: original image - `image001.jpg`, segmented image - `image001_seg.png`. 
The weights and structure of the model are saved to be used later. A few notes on the model training:
    * adding Dropout layers improved the quality of segmentation
    * running the model fitting for a large number of epochs, like 500, greatly improves the accuray of segmentation, even though the loss stays constant
    * the Adam optimizer performed much better than Adadelta


3. The `video_segmentation.py` script takes in a video, splits it into frames and applies the autoencoder on each frame to segment the images. It then converts the segmented images back to a video. To run it, change the `vid` variable to point to the path of the video, which you want to segment. 

The example below shows a schema of an image, not in the training or testing set, being fed through the Autoencoder, and the result. The result is a bit noisy, but this is expected, given the relatively small training data available, which was not fully consistent (I trained the model on images from both highways and city roads, which means the model saw a range of objects around the road and cars). But the main objects (road and cars) are segmented more clearly, with some features, like the tyres being also distinguishable. 
![alt](/segmentation_example.png?raw=true )

With more data and longer training, one can achieve better results. 
