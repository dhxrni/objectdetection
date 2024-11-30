import cv2
import matplotlib.pyplot as plt

# Sets a threshold value (minimum confidence) for the model to decide what it is
thres = 0.50

# Capture the video from the webcam (0 is typically the default camera, change to 1 or another if needed)
cap = cv2.VideoCapture(0)
cap.set(3, 1850)  # Width
cap.set(4, 1850)  # Height
cap.set(10, 70)   # Brightness

# Read class names from coco.names
classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load configuration and weights for the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Configuration file
weightsPath = 'frozen_inference_graph.pb'  # Pre-trained model weights

# Load the neural network
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set input dimensions for the model
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))  # Mean subtraction for normalization
net.setInputSwapRB(True)

# Create the figure and axis for Matplotlib
plt.ion()  # Turn on interactive mode for live updates
fig, ax = plt.subplots()

# Start capturing the video and detecting objects based on the threshold and confidence value
while True:
    success, img = cap.read()
    if not success:
        break  # Exit the loop if the video capture fails

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    # If there are any detected objects
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 <= classId - 1 < len(classNames):
                # Draw rectangle and add class name and confidence score
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR image to RGB (matplotlib uses RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the result using matplotlib in the same figure window
    ax.clear()  # Clear the previous image to update with the new one
    ax.imshow(img_rgb)
    ax.axis('off')  # Hide axes for better display
    plt.draw()  # Update the figure window with the new image

    # Pause briefly to update the frame rate
    plt.pause(0.01)

    # Check if the window was closed by the user
    if not plt.fignum_exists(fig.number):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
