# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from time import sleep
import matplotlib.pyplot as plt


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

rectangles = []
keys = []
#initialize rectangles
while(1):
    _, frame = cap.read()
    #frame = cv2.imread('C:\\Users\\cshu\\Documents\\shool_work\\2023-2024\\sem1\\452\\project\\testHand\\keys.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Gray", gray)
    cv2.imshow("Binary", thresh)
    #plt.imshow(gray)
    #plt.show()
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list for the black rectangles
    black_rectangles = []
    #filter small noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon is a rectangle
        if len(approx) == 4:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Check if the center of the rectangle is black
            center_value = thresh[y + h // 2, x + w // 2]
            if center_value == 255:
                # Add the rectangle to the list
                black_rectangles.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
    """
    #find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #filter small noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    #remove largest contour, removes outline of keys
    areas = [cv2.contourArea(cnt) for cnt in contours]
    if areas: #only delete if nonempty
        largest_contour_index = np.argmax(areas)
        del contours[largest_contour_index]

    #reset keys
    keys = []

    """
    b_mean = cv2.bilateralFilter(thresh,9,75,75)
    blackContours, _ = cv2.findContours(b_mean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #filter small noise
    blackContours = [cnt for cnt in blackContours if cv2.contourArea(cnt) > 500]
    cv2.imshow("Bilatteral", b_mean)
    black_keys = []
    for cnt in blackContours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon is a rectangle
        if len(approx) == 4:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Add the rectangle to the list
            black_keys.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
    """
    # Iterate over each contour
    for i, cnt in enumerate(contours):
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        keys.append(approx)
    
    #Now also find the black keys
    # Find contours in the thresholded image
    #use diff thresh
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("BlackKeyBinary", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area to remove small noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    for approx in keys:
        #draw the white keys
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    # Iterate over the contours
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Store the polygon as a key
        keys.append(approx)

        # Draw the polygon on the frame for preview
        cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
    
        
    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    
# Calculate the x-coordinates of the centroids of the keys
x_centroids = [np.mean(key[:, :, 0]) for key in keys]
# Create a list of tuples where each tuple is (x_centroid, key)
keys_with_x_centroids = list(zip(x_centroids, keys))
# Sort the list of tuples by the x-coordinate of the centroid
keys_with_x_centroids.sort()
# Update the keys list with the sorted keys
keys[:] = [key for x_centroid, key in keys_with_x_centroids]



while True:
    # Read each frame from the webcam
    #sleep(1)
    _, frame = cap.read()

    height, width, c = frame.shape

    # Flip the frame vertically
    #frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    for key in keys:
        cv2.drawContours(frame, [key], -1, (0, 255, 0), 2)
        #xKey, yKey, wKey, hKey = key
        #cv2.rectangle(frame, (xKey, yKey), (xKey + wKey, yKey + hKey), (0, 255, 0), 2)

    for i, key in enumerate(keys):
        # Calculate the centroid of the key
        M = cv2.moments(key)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Write the index of the key on the image
        cv2.putText(frame, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
            
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * width)
                lmy = int(lm.y * height)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            index_finger_tip = handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip_x = int(index_finger_tip.x * width)
            index_finger_tip_y = int(index_finger_tip.y * height)
            # draw a box around the finger tip
            cv2.rectangle(frame, (index_finger_tip_x - 10, index_finger_tip_y - 10), (index_finger_tip_x + 10, index_finger_tip_y + 10), (0, 0, 225), 2)

            #print(f"Index Finger Tip Position: ({index_finger_tip_x}, {index_finger_tip_y}, {index_finger_tip.z})")
            for i, key in enumerate(keys):
                # Use cv2.pointPolygonTest to check if the index finger tip is inside the key
                inside = cv2.pointPolygonTest(key, (index_finger_tip_x, index_finger_tip_y), False) >= 0

            #for i, (xKey, yKey, wKey, hKey) in enumerate(keys):
                #inside = xKey <= index_finger_tip_x <= xKey + wKey and yKey <= index_finger_tip_y <= yKey + hKey
                #print(f"Checking Key {i} at {xKey}, {yKey}, {wKey}, {hKey}")
                if inside:
                    print(f"INSIDE Key {i}")
                    

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
