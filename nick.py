import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('yolov5', 'yolov5s', source='local')

img = 'https://ultralytics.com/images/zidane.jpg'

results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()

results.render()

import pyautogui
import cv2
import numpy as np

# Loop over the frames
while True: 
    # Take a screenshot 
    screen = pyautogui.screenshot()
    # Convert the output to a numpy array
    screen_array = np.array(screen)
    
    # Crop out the region we want - height, width, channels   
    cropped_region = screen_array[315:1880, 2350:3390, :]


    # Convert the color channel order
    corrected_colors = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR)
    
    # Make detections 
    results = model(corrected_colors)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))

    # Cv2.waitkey
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
# Close down the frame
cv2.destroyAllWindows()