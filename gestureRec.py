# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from time import sleep


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('C:\\Users\\cshu\\Documents\\shool_work\\2023-2024\\sem1\\452\\project\\testHand\\hand-gesture-recognition-code\\mp_hand_gesture')

# Load class names
f = open('C:\\Users\\cshu\\Documents\\shool_work\\2023-2024\\sem1\\452\\project\\testHand\\hand-gesture-recognition-code\\gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)

rectangles = []
#initialize rectangles
while(1):
    _, frame = cap.read()

    x, y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get black regions
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    rectangles.clear()

    # Iterate over the contours
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # If the polygon has 4 vertices, it is a rectangle
        if len(approx) == 4:
            rectangles.append(approx)

    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
    
    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) == ord('q'):
        break

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

            # Get index finger tip landmark
            # Get finger tip landmarks
            fingers = ['THUMB', 'INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']
            for finger in fingers:
                finger_tip = handslms.landmark[getattr(mpHands.HandLandmark, f'{finger}_TIP')]
                finger_tip_x = int(finger_tip.x * x)
                finger_tip_y = int(finger_tip.y * y)

                # Print finger tip position
                print(f"{finger} Tip Position: ({finger_tip_x}, {finger_tip_y})")
            
            # Iterate over the contours
            for approx in rectangles:
                
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                for finger in fingers:
                    finger_tip = handslms.landmark[getattr(mpHands.HandLandmark, f'{finger}_TIP')]
                    finger_tip_x = int(finger_tip.x * x)
                    finger_tip_y = int(finger_tip.y * y)

                    # Check if the point is inside the rectangle
                    inside = cv2.pointPolygonTest(approx, (finger_tip_x, finger_tip_y), False) >= 0

                    print(f"{finger} Tip Inside Rectangle: {inside}")


    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
