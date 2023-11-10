# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from time import sleep
from statistics import mode
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
    #_, frame = cap.read()
    frame = cv2.imread('C:\\Users\\cshu\\Documents\\shool_work\\2023-2024\\sem1\\452\\project\\testHand\\keysImg.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow("Binary", thresh)


    ##------------------finding the outer border largest rectangle----------
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

        """epsilon = 0.02 * cv2.arcLength(largest_rectangle, True)
        borderApprox = cv2.approxPolyDP(largest_rectangle, epsilon, True)
        corners = borderApprox.reshape(-1, 2)
        print(corners)
        plt.imshow(thresh)

        # Assign the corners to variables
        bottom_right, top_right, top_left, bottom_left = corners

        x, y, w, h = cv2.boundingRect(largest_rectangle)

        # Compute the perspective transform matrix
        src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
        dst_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(thresh, M, (w, h))
        cv2.imshow("Warped", warped)
        warpCopy = warped

        """# Compute the perspective transform matrix
        rect = cv2.minAreaRect(largest_rectangle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        # Ensure the longer side of the rectangle is horizontal
        if width < height:
            width, height = height, width
            src_pts = np.array([[box[1]], [box[2]], [box[3]], [box[0]]], dtype="float32")
        else:
            src_pts = box.astype("float32")

        #src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(thresh, M, (width, height))
        cv2.imshow("Warped", warped)
        warpCopy = warped
        #"""

        #----------------Find solid black rectangles (black keys)---------------------
        inv_M = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # Use morphological operations to remove the lines
        kernel = np.ones((7,7),np.uint8)
        thresh = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel)

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

        keys = []
        black_keys = [] #these are warped
        for cnt in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            #keys.append(approx)
            #approximately a rectangle
            if len(approx) == 4:
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

                # Check the aspect ratio of the bounding rectangle
                aspect_ratio = float(w)/h
                #if 0.6 <= aspect_ratio <= 0.15:
                #    # Store the polygon as a key
                #    keys.append(approx)
                #
                #    #cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
                #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            
        cv2.imshow("Warp copy", warpCopy)

        ##-----------------find white keys-----------------
        num_polygons = 14 #CHANGE THIS VAL LATER, INFER FROM NUM OF BLACK KEYS
        polygons = []
        #split border into polygons (white keys)
        for i in range(num_polygons):
            x1 = i * width // num_polygons
            x2 = (i + 1) * width // num_polygons
            polygon = np.array([[x1, 0], [x2, 0], [x2, height-1], [x1, height-1]], dtype="int")
            polygons.append(polygon)

        #create mask to fill the smaller rectangles (black keys)
        mask = np.zeros_like(warpCopy)
        for curr_key in black_keys:
            # Get the bounding rectangle of the contour
            x, y, w, h = curr_key#cv2.boundingRect(cnt)
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
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                transformed_approx = cv2.perspectiveTransform(approx.reshape(-1,1,2).astype('float32'), inv_M)
                # Reshape back to original shape and convert to int
                transformed_approx = transformed_approx.reshape(-1,1,2).astype(int)

                keys.append(transformed_approx)
                cv2.drawContours(frame, [transformed_approx],  -1, (0, 0, 255), 2)
        

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
x_centroids = [np.mean(key[:, 0, 0]).tolist() for key in keys]
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
