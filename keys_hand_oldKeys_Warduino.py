"""""
# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import serial
import tensorflow as tf
import time
from keras.models import load_model
from time import sleep

time.sleep(1)
arduino = serial.Serial(port='COM6', baudrate=9600, timeout=.1) 

def write_read(x): 
	arduino.write(bytes(x,'utf-8'))
	time.sleep(1) 
	data = arduino.readline().decode().strip()
	return data 

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Binary", thresh)
    # Convert the image to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get only the paper
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

    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) == ord('p'):
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
                    #print(int.from_bytes(val,byteorder='big'))
                    
                    

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
"""

# import necessary packages
import cv2
import serial
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from time import sleep

time.sleep(1)
arduino = serial.Serial(port='COM7', baudrate=38400, timeout=3) 

def write_read(x): 
    arduino.write(bytes(x, 'utf-8'))
    data = arduino.readline()
    return data 

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
   # _, frame = cap.read()
    frame = cv2.imread('keys.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Binary", thresh)
    # Convert the image to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """# Threshold the image to get only the paper
    _, paper_mask = cv2.threshold(gray, 180, 127, cv2.THRESH_BINARY)

    # Bitwise-and the mask with the original image
    paper_only = cv2.bitwise_and(frame, frame, mask=paper_mask)
    cv2.imshow("paper only", paper_only)
    # Convert the paper-only image to grayscale
    gray_paper = cv2.cvtColor(paper_only, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get only the keys
    _, thresh = cv2.threshold(gray_paper, 70, 255, cv2.THRESH_BINARY_INV)
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


i = 0
while True:
    i=i+1
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
            
            thumb_tip = handslms.landmark[mpHands.HandLandmark.THUMB_TIP]
            thumb_tip_x = int(thumb_tip.x * width)
            thumb_tip_y = int(thumb_tip.y * height)

            index_finger_tip = handslms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip_x = int(index_finger_tip.x * width)
            index_finger_tip_y = int(index_finger_tip.y * height)
            
            middle_finger_tip = handslms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_tip_x = int(middle_finger_tip.x * width)
            middle_finger_tip_y = int(middle_finger_tip.y * height)
            
            ring_finger_tip = handslms.landmark[mpHands.HandLandmark.RING_FINGER_TIP]
            ring_finger_tip_x = int(ring_finger_tip.x * width)
            ring_finger_tip_y = int(ring_finger_tip.y * height)
            
            pinky_tip = handslms.landmark[mpHands.HandLandmark.PINKY_TIP]
            pinky_tip_x = int(pinky_tip.x * width)
            pinky_tip_y = int(pinky_tip.y * height)
            
            # draw a box around the finger tip
            cv2.rectangle(frame, (thumb_tip_x - 10, thumb_tip_y - 10), (thumb_tip_x + 10, thumb_tip_y + 10), (0, 0, 225), 2)
            cv2.rectangle(frame, (index_finger_tip_x - 10, index_finger_tip_y - 10), (index_finger_tip_x + 10, index_finger_tip_y + 10), (0, 0, 225), 2)
            cv2.rectangle(frame, (middle_finger_tip_x - 10, middle_finger_tip_y - 10), (middle_finger_tip_x + 10, middle_finger_tip_y + 10), (0, 0, 225), 2)
            cv2.rectangle(frame, (ring_finger_tip_x - 10, ring_finger_tip_y - 10), (ring_finger_tip_x + 10, ring_finger_tip_y + 10), (0, 0, 225), 2)
            cv2.rectangle(frame, (pinky_tip_x - 10, pinky_tip_y - 10), (pinky_tip_x + 10, pinky_tip_y + 10), (0, 0, 225), 2)

            key_write_arr = np.array([0,0,0,0,0])

            #print(f"Index Finger Tip Position: ({index_finger_tip_x}, {index_finger_tip_y})")
            for t, key in enumerate(keys):
                # Use cv2.pointPolygonTest to check if the index finger tip is inside the key
                thumb_inside = cv2.pointPolygonTest(key, (thumb_tip_x, thumb_tip_y), False) >= 0
        
            #for i, (xKey, yKey, wKey, hKey) in enumerate(keys):
                #inside = xKey <= index_finger_tip_x <= xKey + wKey and yKey <= index_finger_tip_y <= yKey + hKey
                #print(f"Checking Key {i} at {xKey}, {yKey}, {wKey}, {hKey}")
                if thumb_inside:
                    #print(f"Thumb Inside Key {t}")
                    thumb_key_conversion = t+72
                    key_write_arr[0] = thumb_key_conversion
                    #print("Thumb Note: ")
                    #print(thumb_key_conversion)
                
            for i, key in enumerate(keys):
                index_inside = cv2.pointPolygonTest(key, (index_finger_tip_x, index_finger_tip_y), False) >= 0
                
                if index_inside:
                    #print(f"Index Inside Key {i}")
                    index_key_conversion = i+72
                    key_write_arr[1] = index_key_conversion
                    #print("Index Note: ")
                    #print(index_key_conversion)
                
            
            for m, key in enumerate(keys):
                middle_inside = cv2.pointPolygonTest(key, (middle_finger_tip_x, middle_finger_tip_y), False) >= 0
                
                if middle_inside:
                    #print(f"Middle Inside Key {m}")
                    middle_key_conversion = m+72
                    key_write_arr[2] = middle_key_conversion
                    #print("Middle Note: ")
                    #print(middle_key_conversion)
                
            
            for r, key in enumerate(keys):
                ring_inside = cv2.pointPolygonTest(key, (ring_finger_tip_x, ring_finger_tip_y), False) >= 0
                
                if ring_inside:
                    #print(f"Ring Inside Key {r}")
                    ring_key_conversion = r+72
                    key_write_arr[3] = ring_key_conversion
                    #print("Ring Note: ")
                    #print(ring_key_conversion)
                    
                    
            for p, key in enumerate(keys):
                pinky_inside = cv2.pointPolygonTest(key, (pinky_tip_x, pinky_tip_y), False) >= 0
                
                if pinky_inside:
                    #print(f"Pinky Inside Key {i}")
                    pinky_key_conversion = p+72
                    key_write_arr[4] = pinky_key_conversion
                    #print("Pinky Note: ")
                    #print( pinky_key_conversion)
            j = 0
            
            #write_array = bytearray(key_write_arr) 
            #print("Key Write Array: ", key_write_arr)
            #write_read(write_array)
            key_write_string = str(key_write_arr[0]) + 'n' + str(key_write_arr[1]) + 'n' + str(key_write_arr[2]) + 'n' + str(key_write_arr[3]) + 'n' + str(key_write_arr[4]) + 'n' + 'A'
            print(key_write_string)
            val = write_read(key_write_string)
            print("Received Value: ",val)
            
            '''
            readVal = np.array([0,0,0,0,0])
            while j < 5:
                print("J: ",j)
                if j==5:
                    break
                val = write_read(str(key_write_arr[j]))
                print(val)
                key_write_arr[j]=0
                j = j+1
            print(readVal)'''
                    

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()