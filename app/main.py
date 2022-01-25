import streamlit as st
from PIL import Image
import cv2
import numpy as np
import uvicorn
from collections import Counter

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()


def predict(img1):
    img = np.asarray(img1)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    img = np.asarray(img1)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                c_x = int(det[0] * width)
                c_y = int(det[1] * height)
                w = int(det[2] * width)
                h = int(det[3] * height)
                x = int(c_x - w / 2)
                y = int(c_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    classes_detected = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        classes_detected = classes_detected + [str(classes[class_ids[i]])]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.waitKey()
    return img, classes_detected


st.text(" \n")
st.text(" \n")

st.title('Some classes our model can predict :')
st.text(" \n")
st.text(' - Person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe ...')
st.text(' - Bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light ...')
st.text(' - Chair, sofa, potted plant, bed, dining table, toilet ...')
st.text(' - Tv monitor, laptop, mouse, remote, keyboard, cell phone, toaster, refrigerator ...')
st.text(' - Sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket ...')
st.text(" \n")
st.markdown('  ### ** It\'s can predict over 80 different objects **.')

st.text(" \n")
st.text(" \n")


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.markdown(' # ** Upload an image and let our AI agent predict objects : ** ')
    st.text(" \n")
    st.text(" \n")
    choice = st.sidebar.selectbox("Menu", ["Home"])

    if choice == "Home":
        image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            img1 = load_image(image_file)
            st.image(img1)
            img, Classes_det = predict(img1)
            st.image(img)
            dict = Counter(Classes_det)
            st.markdown(' ## ** We detected the following object ** ')
            st.text(" \n")
            for key, value in dict.items():
                st.markdown(' ** '+ str(value)+' ' +str(key) +' ** ')

    else:
        st.subheader("About")

app = main()
if __name__ == '__main__':
    try:
        uvicorn.run(app, port=8083)
    except SystemExit as se:
        pass


