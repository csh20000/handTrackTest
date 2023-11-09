# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from time import sleep
from statistics import mode


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
    #frame = cv2.imread('C:\\Users\\cshu\\Documents\\shool_work\\2023-2024\\sem1\\452\\project\\testHand\\keysImg.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow("Binary", thresh)


    #dilation to connect the black rectangles with the outer black rectangle
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 2)

    cv2.imshow("Dialated", dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_rectangle = None
    largest_area = 0

    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon is a rectangle
        if len(approx) == 4:
            area = cv2.contourArea(cnt)

            #update the largest rectangle
            if area > largest_area:
                largest_rectangle = cnt
                largest_area = area
    # Draw the largest rectangle on the frame
    if largest_rectangle is not None:
        cv2.drawContours(frame, [largest_rectangle], -1, (0, 255, 0), 2)

    #----------------Find solid black rectangles (black keys)---------------------
    # Use morphological operations to remove the lines
    kernel = np.ones((7,7),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #filter contours based on area
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 500 and cv2.contourArea(cnt) < 10000)]
    contours.sort(key=cv2.contourArea, reverse=True)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    most_common_area = 0
    if(areas):
        most_common_area = mode(areas)

    #filter contours based on most common area (should be just the black keys remaining)
    contours = [cnt for cnt in contours if 0.7 * most_common_area <= cv2.contourArea(cnt) <= 1.3 * most_common_area]


    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        #approximately a rectangle
        if len(approx) == 4:
             # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


            # Check the aspect ratio of the bounding rectangle
            aspect_ratio = float(w)/h
            #if 0.6 <= aspect_ratio <= 0.15:
            #    # Store the polygon as a key
            #    keys.append(approx)
            #
            #    #cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
            #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    # The other contours are the potential rectangles
    """potential_rectangles = contours[1:]

    for potential_rectangle in potential_rectangles:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(potential_rectangle, True)
        approx = cv2.approxPolyDP(potential_rectangle, epsilon, True)

        # Check if the polygon is a rectangle
        if len(approx) == 4:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(potential_rectangle)

            # Check if the average of 5 pixels around the center of the rectangle is white
            center_values = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    center_values.append(thresh[y + h // 2 + dy, x + w // 2 + dx])
            if np.mean(center_values) <= 60:
                # Draw the rectangle on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    """
                
    """# Threshold the image to get only the paper
    _, paper_mask = cv2.threshold(gray, 180, 127, cv2.THRESH_BINARY)

    # Bitwise-and the mask with the original image
    paper_only = cv2.bitwise_and(frame, frame, mask=paper_mask)
    cv2.imshow("paper only", paper_only)
    # Convert the paper-only image to grayscale
    gray_paper = cv2.cvtColor(paper_only, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get only the keys
    _, thresh = cv2.threshold(gray_paper, 70, 255, cv2.THRESH_BINARY_INV)
    
    
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

    # Iterate over each contour
    for i, cnt in enumerate(contours):
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        keys.append(approx)
        # Draw the polygon on the frame for preview
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        #M = cv2.moments(cnt)
        #cX = int(M["m10"] / M["m00"])
        #cY = int(M["m01"] / M["m00"])
        #cv2.putText(frame, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    #Now also find the black keys
    # Find contours in the thresholded image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area to remove small noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Iterate over the contours
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Store the polygon as a key
        keys.append(approx)

        # Draw the polygon on the frame for preview
        cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
    """

    cv2.imshow("Preview", frame)
    
    #input("Press any key to continue...")
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

    # print(result)
    
    className = ''

    for key in keys:
        cv2.drawContours(frame, [key], -1, (0, 255, 0), 2)
        #xKey, yKey, wKey, hKey = key
        #cv2.rectangle(frame, (xKey, yKey), (xKey + wKey, yKey + hKey), (0, 255, 0), 2)

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

            #print(f"Index Finger Tip Position: ({index_finger_tip_x}, {index_finger_tip_y})")
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
