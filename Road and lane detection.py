import numpy as np
import cv2
import os
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from tensorflow import keras

script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model file relative to the script's directory
model_filename = 'model.h5'
model_path = os.path.join(script_dir, model_filename)

input_filename = 'input.mp4'
input_path = os.path.join(script_dir, input_filename)

model = keras.models.load_model(model_path)

class lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):

    #pre processing the image
    small_img = imresize(image, (80,160,3)) #model_size
    small_img = np.array(small_img)  #eff
    small_img = small_img[None,:,:,:] ##forth dimension where the predictions will be stored

    #feed the image to the CNN model
    prediction = model.predict(small_img)
    prediction = prediction[0]*255 #just keeping the prediction dimension and *255 to use it as image (dark)

    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:  #the recent list store last 5 element
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)  #producing an black and white image where the brighmess of each pixel indicates the probability of this pixel being a part of the lane or not
     # avg of last 5 the brighness indicates if t=its in the line or not
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8) #blank with initializing with zeroes
    # we choose our prediction to be the midddle one so the predictions will be shown as green in the final video
    lane_drown = np.dstack((blanks, lanes.avg_fit , blanks)) #stack the average matrix tha contains the predection between two layers of blanks (creates our rgb image)



    #resize again the image to match the input size
    vid = cv2.VideoCapture(input_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(height)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    width = int(width)
    lane_image = imresize(lane_drown , (height, width, 3))


    #overlay our output image over the input image using addWeighted method from the opencv
    results = cv2.addWeighted(image, 1, lane_image ,1 ,0)

    return results


lanes = lanes()

#selecting the video
vid_input = VideoFileClip(input_path)

#choosing a name for the output video
vid_output = 'lane_detected.mp4'

vid_clip = vid_input.fl_image(road_lines)
vid_clip.write_videofile('lane_detected.mp4',fps=30, threads=1, codec="libx264")

