import numpy as np
import mediapipe as mp
import cv2
import math
import tensorflow as tf

model = tf.keras.models.load_model("./Model/islr_model_latest.h5")
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
offset = 20

cap = cv2.VideoCapture(0)

success, frame = cap.read()
height, width, channel = frame.shape

counter = 0
imgsize = 300
while True:

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:

        landmarks = []
        for hand_lendmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lendmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(
                thickness=1, color=(0, 255, 0)), connection_drawing_spec=mp_drawing.DrawingSpec(thickness=12, color=(0, 255, 0)))
            for landmark in hand_lendmarks.landmark:
                landmarks.append(
                    [int(landmark.x * width), int(landmark.y * height)])

        landmarks = np.array(landmarks)
        X_axis = landmarks[:, 0]
        Y_axis = landmarks[:, 1]
        x1 = int(np.min(X_axis) - offset)
        y1 = int(np.min(Y_axis) - offset)
        x2 = int(np.max(X_axis) + offset)
        y2 = int(np.max(Y_axis) + offset)

        imgBlack = np.zeros((imgsize, imgsize, 3), np.uint8)
        imgCrop = frame[y1: y2, x1: x2]

        h, w, _ = imgCrop.shape

        try:
            aspectRatio = h / w
        except:
            aspectRatio = 1

        if aspectRatio > 1:
            try:
                k = imgsize / h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgsize-wCal)/2)
                imgBlack[0:imgResizeShape[0],
                        wGap:imgResizeShape[1]+wGap] = imgResize
            except:
                pass
        else:
            try:

                k = imgsize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgsize-hCal)/2)
                imgBlack[hGap:imgResizeShape[0]+hGap,
                        0:imgResizeShape[1]] = imgResize
            except:
                pass

        hsv_image = cv2.cvtColor(imgBlack, cv2.COLOR_BGR2HSV)

        # define range of green color in HSV
        lower_green = np.array([25, 52, 72])
        upper_green = np.array([102, 255, 255])

        # Threshold the HSV image to get only green colors
        final_image = cv2.inRange(hsv_image, lower_green, upper_green)

        cv2.imshow("final_image", final_image)

        input_image = cv2.resize(final_image, (64, 64))
        input_image = input_image.reshape(1, 64, 64, 1)

        prediction_array = model.predict(input_image)
        prediction_index = np.argmax(prediction_array, axis=1)
        prediction = labels[prediction_index[0]]

        cv2.putText(frame, prediction, (30, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    cv2.imshow("image", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
