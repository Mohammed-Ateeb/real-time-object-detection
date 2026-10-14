# Real-Time Object Detection (600 classes)

A web app that uses your webcam and a pretrained deep learning model to find and name objects in front of the camera, live, inside the browser. The model knows 600 kinds of object, including bottle, mobile phone, watch, glasses, pen, headphones, mug, laptop and person.

## Project objective

To show how a pretrained Convolutional Neural Network (CNN) can be used for real-time object recognition without training a model, without a server, and without any paid API. The webcam feed is analysed continuously. The app draws a box around every object it recognises and displays the object name and the confidence percentage.

## Folder structure

```
DL Project/
├── index.html            Page structure: video, buttons, prediction area
├── style.css             Styling and responsive layout
├── script.js             Model loading, webcam, image preparation, detection, drawing
├── models/
│   ├── yolov8s-oiv7.onnx  The pretrained model (about 46 MB)
│   └── labels.json        The 601 class names, in class-number order
├── lib/                  ONNX Runtime Web library files, kept locally so no internet is needed
├── export_model.py       Optional: how the model files were produced (Python)
└── README.md             This file
```

## Technologies used

| Technology | Purpose |
|---|---|
| HTML5 | Page structure, the `<video>` element and the `<canvas>` for boxes |
| CSS3 | Layout, styling, responsiveness |
| JavaScript | App logic, image preparation, box decoding |
| YOLOv8s trained on Open Images V7 (Ultralytics) | The pretrained object detection model |
| ONNX / ONNX Runtime Web | Model file format, and the library that runs it in the browser (on the GPU through WebGPU, or on the CPU through WebAssembly) |
| WebRTC `getUserMedia` API | Access to the webcam |
| Python + Ultralytics (one time only) | Converting the trained model to ONNX, see `export_model.py` |

## How the model works

The model is **YOLOv8s** ("You Only Look Once", version 8, small size), a CNN for object detection.

- **Already trained.** Ultralytics trained it on **Open Images V7**: about 1.7 million photos with boxes drawn around objects of **600 categories**. This project does not train anything; `export_model.py` only converts the trained weights from PyTorch format to ONNX so a browser can run them.
- **One look.** The network looks at the whole image in a single forward pass (hence "you only look once") and makes a prediction for **8,400 candidate boxes** spread over the image at three sizes, so both small and large objects can be found.
- **Its layers:**
  - a **backbone** of convolution layers that turns the image into features (edges, then textures, then object parts);
  - a **neck** that combines features from different scales;
  - a **head** that, for every candidate box, outputs the box position and a score for each of the 601 classes.
- **Input:** a 640 × 640 colour image with pixel values between 0 and 1.
- **Output:** a table of 605 × 8,400 numbers: 4 box numbers plus 601 class scores for each candidate box.

**Why not MobileNet?** The project started with MobileNet image classification in TensorFlow.js. It knew 1000 labels but was very inaccurate on a real webcam, and it gives one label for the whole picture. Detection models find each object separately, which works much better on busy webcam images. Earlier versions of this project also tried COCO-SSD and EfficientDet, which only know 80 classes (no watch, glasses or pen).

## How webcam detection works

1. **Load the model** – when the page opens, `labels.json` and the ONNX model are loaded and ONNX Runtime prepares the network, on the GPU if possible. A loading message is shown and the Start button stays disabled until it finishes.
2. **Ask for the camera** – clicking *Start Camera* calls `navigator.mediaDevices.getUserMedia({ video: ... })`. The browser shows a permission pop-up.
3. **Show the feed** – the camera stream is attached to the `<video>` element. A transparent `<canvas>` of the same size sits on top of it for the boxes.
4. **Prepare the frame** (`prepareInput`) – the frame is shrunk to fit 640 × 640 without being squashed (empty strips are filled with grey, called *letterboxing*), and the pixels are turned into numbers between 0 and 1 in the layout the network expects.
5. **Run the network** (`session.run`) – one forward pass gives the 8,400 candidate boxes.
6. **Decode the output** (`detectObjects`) – for every candidate, take the class with the highest score; keep candidates scoring at least 35%; convert their boxes back to the camera frame's pixels.
7. **Remove duplicates** (`nonMaximumSuppression`) – the same object is usually found by several overlapping boxes. Starting from the most confident, a box is kept only if it overlaps less than 45% (*intersection over union*) with a box of the same class that was already kept.
8. **Filter and steady the result** – if *Hide people and clothing* is ticked (the default), classes such as Person, Man, Human face and Clothing are removed, because they are almost always in view when you hold an object up. Then `steadyDetections` keeps an object on screen for 1 second after it was last seen, so a box does not flicker off when the model misses it for a single frame.
9. **Update the page** – boxes and labels are drawn on the canvas; the side panel shows the most confident object and its confidence, with other objects listed below.
10. **Repeat** until *Stop Camera* is clicked, which stops the camera tracks and ends the loop.

Errors are handled with clear messages: permission denied, no camera found, camera that cannot be opened, and model loading failure.

## Model test

The model was run inside this app on 42 photos from Wikimedia Commons. A photo counts as correct when the expected class is found with at least 35% confidence (the app's threshold).

| Object | Correct |
|---|---|
| Bottle | 7 / 12 |
| Mobile phone | 4 / 12 |
| Mug / coffee cup | 5 / 6 |
| Watch | 8 / 12 |

This is a small informal test, not a benchmark. Several of the phone and bottle photos show the object very small, in a crowd, or not at all, so those rows understate the model. The 80-class models tried earlier could not detect watches at all.

## How to run the project

The model is loaded from the `models` folder, and browsers only allow camera access on `https://` or `http://localhost`, so run the project from a local web server instead of double-clicking `index.html`.

**Option 1 – VS Code Live Server (easiest)**

1. Install the *Live Server* extension in VS Code.
2. Right-click `index.html` and choose **Open with Live Server**.

**Option 2 – Python**

```
cd "DL Project"
python -m http.server 8080
```

Then open <http://localhost:8080> in the browser. If that port is busy, use any other free number in both places.

Then:

1. Wait for the message "Model loaded (601 object classes)".
2. Click **Start Camera** and allow camera access.
3. If the wrong camera is shown (for example a linked phone instead of the laptop webcam), pick the right one in the **Camera** dropdown. The choice is remembered.
4. Show objects to the camera. Untick **Hide people and clothing** if you want people to be detected too.
5. Click **Stop Camera** when finished.

Use Chrome or Edge for the best speed (they support WebGPU). Other browsers fall back to the CPU, which is several times slower. No internet connection is needed: the model and the library are both in the project folder.

## Limitations

- **Slow on weak hardware.** On a laptop with integrated graphics the boxes update about 2 times per second (the video itself stays smooth). Without WebGPU it is slower.
- **Lower confidence scores.** With 600 classes the model spreads its confidence over more choices, so correct answers often score 40–70% rather than 90%+.
- **Similar classes.** Open Images has overlapping labels (Person / Man / Woman, Mug / Coffee cup, Bottle / Vase), so the name shown can be a near neighbour of the one you expect.
- **Sensitive to conditions.** Poor lighting, unusual angles, and small or partly hidden objects reduce accuracy. Hold objects large in the frame.
- **Confidence is not certainty.** A high percentage can still be wrong.
- **Slow first load.** The model (about 46 MB) and the library (about 22 MB) are prepared when the page opens, which can take 15–30 seconds.
- **Boxes can lag slightly.** Because an object stays on screen for 1 second after it was last seen, a box can briefly remain after the object is moved away.

## Possible future improvements

- Use the medium model (`yolov8m-oiv7`) on computers with a strong GPU for higher accuracy
- Show the current frames per second and whether the GPU is in use
- Counting objects of each type
- Fine-tuning the model on photos of specific objects
