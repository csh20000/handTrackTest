


# import necessary packages
#import cv2
import numpy as np
#import mediapipe as mp
#import tensorflow as tf
from keras.models import load_model
from time import sleep
from statistics import mode
import matplotlib.pyplot as plt
from collections import deque
from picamera2 import Picamera2, Preview
import serial


sleep(1)

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=38400, timeout=3) 

def write_read(x): 
    arduino.write(bytes(x, 'utf-8'))
    data = arduino.readline()
    return data 

# Initialize the webcam
# cap = cv2.VideoCapture(0)
picam = Picamera2()

config = picam.create_preview_configuration()
picam.configure(config)

#START PREVIEW IS THE ISSUE
# picam.start_preview(Preview.QTGL)

preview_config = picam.create_preview_configuration(main={"size": (640, 480),"format":"BGR888"})
#capture_config = picam.create_still_configuration(main={"size": (640, 480)})

picam.configure(preview_config)
picam.start()
import mediapipe as mp
import tensorflow as tf
import cv2

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

rectangles = []
keys = []
#initialize rectangles

#----Threshold slider-----
thold_val = 100 #threshold for key border key border
tWind = 'TrackBar Window'
cv2.namedWindow(tWind)

def onThresh(val):
    global thold_val
    thold_val = val

cv2.createTrackbar('Thresh', tWind, thold_val, 255, onThresh)
#-------------------------

kernel_size_bk = 5
def onThresh(val):
    global kernel_size_bk
    kernel_size_bk = val

cv2.createTrackbar('Size', tWind, kernel_size_bk, 30, onThresh)

while(1):
    frame = picam.capture_array("main")
    #frame = cv2.imread('C:\\Users\\cshu\\Documents\\shool_work\\2023-2024\\sem1\\452\\project\\testHand\\keysImg.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    equ = cv2.equalizeHist(gray)
    blur = cv2.bilateralFilter(equ,9,75,75)


    #GLOBAL THRESH
    _, thresh = cv2.threshold(gray, thold_val, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Binary", thresh)


    ##------------------finding the outer border largest rectangle----------
    #dilation to connect the black rectangles with the outer black rectangle
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 2)

    cv2.imshow("Dialated", dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_rectangle = None
    largest_area = 0
    largest_approx = None

    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon is a rectangle
        if len(approx) == 4:
            area = cv2.contourArea(cnt)

            #update the largest rectangle
            if area > largest_area:
                largest_rectangle = cnt
                largest_area = area
                largest_approx = approx

    # Draw the largest rectangle on the frame
    if largest_rectangle is not None:
        cv2.drawContours(frame, [largest_rectangle], -1, (0, 255, 0), 2)


        corners = largest_approx.reshape(-1, 2)
        print(corners)

        # Sort corners based on the sum of x and y
        corners = sorted(corners, key=lambda corner: corner[0] + corner[1])

        # Assign corners based on the sum of x and y
        top_left, _ , _, bottom_right = corners

        # Assign the remaining corners based on y value
        if corners[1][1] < corners[2][1]:
            top_right, bottom_left = corners[1], corners[2]
        else:
            top_right, bottom_left = corners[2], corners[1]

        _, _, border_w, border_h = cv2.boundingRect(largest_rectangle)

        top_left = [top_left[0] + 5, top_left[1] + 5]
        bottom_left = [bottom_left[0] + 5, bottom_left[1] - 5]
        top_right = [top_right[0] - 5, top_right[1] + 5]
        bottom_right = [bottom_right[0] - 5, bottom_right[1] - 5]

        # Compute the perspective transform matrix
        src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
        dst_pts = np.float32([[0, 0], [border_w-1, 0], [border_w-1, border_h-1], [0, border_h-1]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(thresh, M, (border_w, border_h))
        #c##v2.imshow("Warped", warped)
        warpCopy = warped

        #----------------Find solid black rectangles (black keys)---------------------
        inv_M = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # Use morphological operations to remove the lines
        kernel = np.ones((kernel_size_bk,kernel_size_bk),np.uint8)
        blackThresh = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)

        #cv2.imshow("BlackKeyThresh", blackThresh)

        #cv2.findContours(blackThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(blackThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #filter contours based on area
        contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 500)]
        keys = []
        black_keys = [] #these are warped

        for cnt in contours:
            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            cv2.drawContours(frame, [approx],  -1, (0, 255, 255), 2)

            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(warpCopy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rect = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
            #inv_M = cv2.getPerspectiveTransform(dst_pts, src_pts) define this above
            inv_rect = cv2.perspectiveTransform(rect.reshape(-1,1,2), inv_M)
            inv_rect = inv_rect.astype(int)

            keys.append(inv_rect)
            black_keys.append([x,y,w,h])

            # Draw the rectangle on the original frame
            cv2.polylines(frame, [inv_rect], True, (255, 0, 0), 2)

            
        #cv2.imshow("Warp copy", warpCopy)

        
        ##-----------------find white keys-----------------
        #value to not pass calibration if black keys not a multiple of 5
        valid_black_keys = False
        if len(black_keys) % 5 == 0:
            valid_black_keys = True

        num_polygons = 7*len(black_keys)//5 #7 white keys per 5 black keys
        polygons = []
        #split border into polygons (white keys)
        for i in range(num_polygons):
            x1 = i * border_w // num_polygons
            x2 = (i + 1) * border_w // num_polygons
            polygon = np.array([[x1, 0], [x2, 0], [x2, border_h-1], [x1, border_h-1]], dtype="int")
            polygons.append(polygon)

        #create mask to fill the smaller rectangles (black keys)
        mask = np.zeros_like(warpCopy)
        for curr_key in black_keys:
            # Get the bounding rectangle of the contour
            x, y, w, h = curr_key
            # Fill the rectangle on the mask
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        #subtract smaller rectangles from larger polygons (white - black)
        for polygon in polygons:
            poly_mask = np.zeros_like(warpCopy)
            cv2.fillPoly(poly_mask, [polygon], (255, 255, 255))
            poly_mask = cv2.subtract(poly_mask, mask)

            # Find contours in the polygon mask
            poly_contours, _ = cv2.findContours(poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Approximate the contours to polygons and add them to keys
            for cnt in poly_contours:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                transformed_approx = cv2.perspectiveTransform(approx.reshape(-1,1,2).astype('float32'), inv_M)
                # Reshape back to original shape and convert to int
                transformed_approx = transformed_approx.reshape(-1,1,2).astype(int)

                keys.append(transformed_approx)
                cv2.drawContours(frame, [transformed_approx],  -1, (0, 0, 255), 2)

    cv2.imshow("Preview", frame)
    
    #input("Press any key to continue...")
    if cv2.waitKey(1) == ord('q') and valid_black_keys:
        break

    
# Calculate the x-coordinates of the centroids of the keys
x_centroids = [np.mean(key[:, 0, 0]).tolist() for key in keys]
# Create a list of tuples where each tuple is (x_centroid, key)
keys_with_x_centroids = list(zip(x_centroids, keys))
# Sort the list of tuples by the x-coordinate of the centroid
keys_with_x_centroids.sort()
# Update the keys list with the sorted keys
keys[:] = [key for x_centroid, key in keys_with_x_centroids]



#z_diffs = deque(maxlen=100)
#y_diffs = deque(maxlen=100)

# Initialize the plot
#plt.ion()
#fig, ax = plt.subplots()
#line1, = ax.plot(z_diffs, color='blue')
#line2, = ax.plot(y_diffs, color='red')
#previous_y = None

while True:
    # Read each frame from the webcam
    frame = picam.capture_array("main")

    height, width, c = frame.shape

    # Flip the frame vertically
    #frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = framergb

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
                    #print(middle_key_conversion
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
                    

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
