import numpy as np
import mediapipe as mp
import cv2
import math
import io
import ssl
from gtts import gTTS
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, request, redirect, stream_with_context, jsonify, url_for, send_file
app = Flask(__name__, template_folder='HTML')


model = load_model("./Model/islr_model_latest.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " "]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
offset = 20

counter = 0
imgsize = 300

cap = None
width = 640
height = 480

def get_content():
    success, frame = cap.read()

    if not success:
        return None
    else:
        frame = cv2.flip(frame, 1)
        print(frame)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        if results.multi_hand_landmarks:

            landmarks = []
            for hand_lendmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lendmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1,color=(0,255,0)), connection_drawing_spec=mp_drawing.DrawingSpec(thickness=12,color=(0,255 , 0)))
                for landmark in hand_lendmarks.landmark:
                    landmarks.append([int(landmark.x * width), int(landmark.y * height)])

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
                    imgBlack[0:imgResizeShape[0], wGap:imgResizeShape[1]+wGap] = imgResize
                except:
                    pass
            else:
                try:
                    k = imgsize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgsize-hCal)/2)
                    imgBlack[hGap:imgResizeShape[0]+hGap, 0:imgResizeShape[1]] = imgResize
                except:
                    pass

            hsv_image = cv2.cvtColor(imgBlack, cv2.COLOR_BGR2HSV)

            # define range of green color in HSV
            lower_green = np.array([25,52,72])
            upper_green = np.array([102,255,255])

            # Threshold the HSV image to get only blue colors
            final_image = cv2.inRange(hsv_image, lower_green, upper_green)

            input_image = cv2.resize(final_image, (64, 64))
            input_image = input_image.reshape(1, 64, 64, 1)

            prediction_array = model.predict(input_image, verbose=1)
            prediction_index = np.argmax(prediction_array, axis=1)
            prediction = labels[prediction_index[0]]

            return (frame, prediction)
        else:
            return (frame, "")

def send_img():
    while True:
        frame = get_content()

        if frame is None:
            break
        else:
            frame = frame[0]
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/SLR_LAB')
def slr_lab():
    return render_template('slrlab.html')

@app.route('/open', methods=['GET'])
def open():
    isPermissionAllowed = (request.args.get("isPermissionAllowed") == "true")
    if isPermissionAllowed:
        return redirect(url_for('slr_lab', isCameraOpen='true'))
    else:
        return redirect(url_for('slr_lab'))


@app.route('/close', methods=['GET'])
def close():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    cap = None
    return redirect(url_for('slr_lab'))

@app.route('/video')
def video():
    global cap
    cap = cv2.VideoCapture(0)
    return Response(send_img(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pridiction')
def pridiction():
    if cap is not None:
        label = get_content()

        if label is None:
            return {"label" : ""}
        else:
            return {"label" : label[1]}
    else:
        return redirect(url_for('slr_lab'))
    

@app.route("/download", methods=['POST'])
def download_mp3():
    text = (request.get_json())['text']
    mp3 = io.BytesIO()
    tts = gTTS(text)
    tts.write_to_fp(mp3)
    mp3.seek(0)
    return send_file(mp3, as_attachment=True, download_name='audio.mp3')
    
@app.route('/technology')
def technology():
    return render_template('technology.html')



@app.route('/help')
def help_section():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True)