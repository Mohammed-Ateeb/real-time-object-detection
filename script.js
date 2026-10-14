// =====================================================================
// Real-Time Object Detection (600 object classes)
// Pretrained YOLOv8 model trained on Google's Open Images dataset,
// run in the browser with ONNX Runtime Web.
// =====================================================================

// ---------- 1. Grab the HTML elements we need ----------
const video = document.getElementById("webcam");
const videoBox = document.getElementById("videoBox");
const overlay = document.getElementById("overlay");
const overlayContext = overlay.getContext("2d");
const placeholder = document.getElementById("placeholder");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const cameraSelect = document.getElementById("cameraSelect");
const statusBox = document.getElementById("status");
const statusText = document.getElementById("statusText");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const otherObjectsEl = document.getElementById("otherObjects");
const hidePeopleBox = document.getElementById("hidePeople");

// ---------- 2. Settings and app state ----------
// The trained model and the list of its 600 class names (in the models folder).
const MODEL_URL = "models/yolov8s-oiv7.onnx";
const LABELS_URL = "models/labels.json";
// The model expects a square 640 x 640 image.
const INPUT_SIZE = 640;
// Detections with a score below this (0.35 = 35%) are ignored.
const MIN_SCORE = 0.35;

// Classes hidden when "Hide people and clothing" is ticked. When you hold
// up an object these are almost always in view and would push the object
// itself down the list.
const PEOPLE_CLASSES = [
  "Person", "Man", "Woman", "Boy", "Girl",
  "Human arm", "Human beard", "Human body", "Human ear", "Human eye", "Human face",
  "Human foot", "Human hair", "Human hand", "Human head", "Human leg", "Human mouth", "Human nose",
  "Clothing", "Footwear", "Jeans", "Dress", "Shirt", "Suit"
];

// An object stays on screen for this long (in milliseconds) after the model
// last saw it. This stops boxes flickering when one frame misses an object.
const HOLD_TIME = 1000;

let session = null;  // the loaded model, ready to run
let labels = [];     // labels[3] is the name of class number 3, and so on
let stream = null;   // the active webcam stream (null when the camera is off)
let shownObjects = []; // objects currently on screen, each with the time it was last seen

// The camera picked in the dropdown. It is remembered between visits.
// Empty means "let the browser choose".
let chosenCameraId = localStorage.getItem("cameraId") || "";

// Small helper to update the status message.
// type is one of: "loading", "ready", "error", "" (neutral)
function setStatus(message, type) {
  statusText.textContent = message;
  statusBox.className = "status " + type;
}

// ---------- 3. Load the pretrained model ----------
// YOLOv8 ("You Only Look Once", version 8) is a Convolutional Neural
// Network (CNN) for object detection. This copy was already trained by
// Ultralytics on Open Images V7: about 1.7 million photos with boxes
// drawn around objects of 600 categories (Bottle, Mobile phone, Watch,
// Glasses, Pen, Headphones, Mug, Laptop, Person...).
// We do NOT train anything here. The trained network was converted to the
// ONNX file format once, and the browser only loads and runs it.
async function loadModel() {
  try {
    labels = await (await fetch(LABELS_URL)).json();
    // "webgpu" runs the network on the graphics card. If the browser does
    // not support it, ONNX Runtime falls back to "wasm" (the CPU).
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["webgpu", "wasm"]
    });
    setStatus("Model loaded (" + labels.length + " object classes). Click \"Start Camera\" to begin.", "ready");
    startBtn.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus("Could not load the model. Make sure the page is opened through a local web server (see README) and refresh.", "error");
  }
}

// ---------- 4. Start the webcam ----------
async function startCamera() {
  // Older browsers, or pages not served from https/localhost, have no camera API.
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("This browser cannot access the camera. Open the page from http://localhost or https in a modern browser.", "error");
    return;
  }

  startBtn.disabled = true;
  setStatus("Requesting camera permission...", "loading");

  try {
    // getUserMedia makes the browser show the permission pop-up.
    // If a camera was picked in the dropdown we ask for exactly that one
    // ("exact" stops the browser from quietly using a different camera),
    // otherwise "video: true" asks for the browser's default camera.
    stream = await navigator.mediaDevices.getUserMedia({
      video: chosenCameraId ? { deviceId: { exact: chosenCameraId } } : true,
      audio: false
    });

    // Connect the camera stream to the <video> element and wait
    // until real frames are available.
    video.srcObject = stream;
    await video.play();
    await fillCameraList();

    // Give the drawing canvas the same pixel size as the video frame and
    // make the box on the page the same shape, so boxes can be drawn in
    // video pixel coordinates.
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    videoBox.style.aspectRatio = video.videoWidth + " / " + video.videoHeight;

    placeholder.classList.add("hidden");
    stopBtn.disabled = false;
    // Show which camera is in use, so a wrong or virtual camera is easy to spot.
    const cameraName = stream.getVideoTracks()[0].label || "unknown camera";
    setStatus("Camera is running (" + cameraName + "). Detecting objects...", "ready");

    detectLoop();
  } catch (error) {
    console.error(error);
    stopCamera();
    if (error.name === "OverconstrainedError" && chosenCameraId) {
      // The remembered camera no longer exists (unplugged, or the browser
      // gave it a new id). Forget it so the next start uses the default.
      chosenCameraId = "";
      localStorage.removeItem("cameraId");
      setStatus("The selected camera is not available any more. Click Start Camera to use the default camera.", "error");
    } else {
      setStatus(getCameraErrorMessage(error), "error");
    }
  }
}

// Fill the dropdown with every camera on this computer (built-in webcam,
// USB webcam, a linked phone, virtual cameras...). Browsers only reveal
// camera names after permission is granted, so this runs after the
// camera has started.
async function fillCameraList() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(function (device) {
    return device.kind === "videoinput";
  });

  cameraSelect.innerHTML = "";
  cameras.forEach(function (camera, index) {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || "Camera " + (index + 1);
    cameraSelect.appendChild(option);
  });

  // Select the camera that is actually running right now.
  cameraSelect.value = stream.getVideoTracks()[0].getSettings().deviceId;
  cameraSelect.disabled = false;
}

// When another camera is picked: remember it, then restart with it.
function switchCamera() {
  chosenCameraId = cameraSelect.value;
  localStorage.setItem("cameraId", chosenCameraId);
  stopCamera();
  startCamera();
}

// Turn the browser's camera error into a message a user can act on.
function getCameraErrorMessage(error) {
  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access in your browser's site settings, then click Start Camera again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found. Connect a webcam and try again.";
    case "NotReadableError":
      return "This camera could not be opened. Close other apps using it (Zoom, Teams, Camera), check its privacy shutter or camera key, and check Windows Settings > Privacy & security > Camera. Or pick another camera below.";
    default:
      return "Could not start the camera: " + error.message;
  }
}

// ---------- 5. Stop the webcam ----------
function stopCamera() {
  if (stream) {
    // Stopping every track releases the camera (the camera light turns off).
    stream.getTracks().forEach(function (track) {
      track.stop();
    });
  }
  // Setting stream to null also ends the detection loop (see below).
  stream = null;
  video.srcObject = null;
  shownObjects = [];
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);

  placeholder.classList.remove("hidden");
  stopBtn.disabled = true;
  startBtn.disabled = (session === null);
  resetResult();
  setStatus("Camera stopped. Click \"Start Camera\" to begin again.", "");
}

// ---------- 6. Prepare an image for the model ----------
// The model needs a 640 x 640 image stored as numbers. Steps:
//   a) "Letterbox": shrink the frame to fit inside 640 x 640 without
//      squashing it, and fill the empty strips with grey.
//   b) Turn every pixel's red, green and blue values (0-255) into
//      numbers between 0 and 1.
//   c) Store all red values first, then all green, then all blue.
//      This [1, 3, 640, 640] layout is what the network was trained with.
const inputCanvas = document.createElement("canvas");
inputCanvas.width = INPUT_SIZE;
inputCanvas.height = INPUT_SIZE;
const inputContext = inputCanvas.getContext("2d", { willReadFrequently: true });

function prepareInput(source, sourceWidth, sourceHeight) {
  // a) Letterbox
  const scale = Math.min(INPUT_SIZE / sourceWidth, INPUT_SIZE / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  const padX = (INPUT_SIZE - width) / 2;
  const padY = (INPUT_SIZE - height) / 2;
  inputContext.fillStyle = "rgb(114, 114, 114)";
  inputContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  inputContext.drawImage(source, padX, padY, width, height);

  // b) and c) Pixels to numbers
  const pixels = inputContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data; // R,G,B,A,R,G,B,A...
  const area = INPUT_SIZE * INPUT_SIZE;
  const numbers = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    numbers[i] = pixels[i * 4] / 255;                // red
    numbers[area + i] = pixels[i * 4 + 1] / 255;     // green
    numbers[2 * area + i] = pixels[i * 4 + 2] / 255; // blue
  }

  // A tensor is the model's input format: the numbers plus their shape.
  const tensor = new ort.Tensor("float32", numbers, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  // scale and padding are needed later to map boxes back onto the frame.
  return { tensor: tensor, scale: scale, padX: padX, padY: padY };
}

// ---------- 7. Run the model on one image ----------
async function detectObjects(source, sourceWidth, sourceHeight) {
  const input = prepareInput(source, sourceWidth, sourceHeight);

  // The forward pass: the network looks at the image once ("you only look
  // once") and makes a prediction for 8400 candidate boxes spread over the
  // whole image, all at the same time.
  const results = await session.run({ [session.inputNames[0]]: input.tensor });
  const output = results[session.outputNames[0]].data;

  // The output is a table of 605 rows x 8400 columns (one column per
  // candidate box):
  //   rows 0-3   : box centre x, centre y, width, height (in the 640 image)
  //   rows 4-604 : the score of each of the 601 classes for that box
  const boxCount = output.length / (4 + labels.length);

  // Step 1: for every candidate box, find its best class and score.
  // (Looping class by class reads the table in memory order, which is fast.)
  const bestScore = new Float32Array(boxCount);
  const bestClass = new Int32Array(boxCount);
  for (let c = 0; c < labels.length; c++) {
    const rowStart = (4 + c) * boxCount;
    for (let b = 0; b < boxCount; b++) {
      if (output[rowStart + b] > bestScore[b]) {
        bestScore[b] = output[rowStart + b];
        bestClass[b] = c;
      }
    }
  }

  // Step 2: keep only the confident boxes and convert them to the original
  // frame's pixels (undo the letterbox).
  const candidates = [];
  for (let b = 0; b < boxCount; b++) {
    if (bestScore[b] < MIN_SCORE) {
      continue;
    }
    const centreX = output[b];
    const centreY = output[boxCount + b];
    const width = output[2 * boxCount + b];
    const height = output[3 * boxCount + b];
    candidates.push({
      name: labels[bestClass[b]],
      score: bestScore[b],
      x: (centreX - width / 2 - input.padX) / input.scale,
      y: (centreY - height / 2 - input.padY) / input.scale,
      width: width / input.scale,
      height: height / input.scale
    });
  }

  // Step 3: remove duplicate boxes for the same object.
  return nonMaximumSuppression(candidates);
}

// Non-maximum suppression (NMS)
// The model usually finds the same object with several overlapping boxes.
// Going from the most confident box down, we keep a box only if it does
// not overlap a lot with an already kept box of the same class.
const MAX_OVERLAP = 0.45;

function nonMaximumSuppression(boxes) {
  boxes.sort(function (a, b) {
    return b.score - a.score; // most confident first
  });
  const kept = [];
  boxes.forEach(function (box) {
    const isDuplicate = kept.some(function (keptBox) {
      return keptBox.name === box.name && overlap(keptBox, box) > MAX_OVERLAP;
    });
    if (!isDuplicate) {
      kept.push(box);
    }
  });
  return kept;
}

// "Intersection over union": the shared area of two boxes divided by the
// area they cover together. 0 = no overlap, 1 = the same box.
function overlap(a, b) {
  const sharedWidth = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const sharedHeight = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const shared = sharedWidth * sharedHeight;
  return shared / (a.width * a.height + b.width * b.height - shared);
}

// ---------- 8. The real-time detection loop ----------
// This is the heart of the project. It repeats for as long as the
// camera stream that started it is still the active one.
async function detectLoop() {
  const myStream = stream;

  while (stream === myStream) {
    let detections = await detectObjects(video, video.videoWidth, video.videoHeight);

    // The camera may have been stopped while the model was running.
    if (stream !== myStream) {
      break;
    }

    if (hidePeopleBox.checked) {
      detections = detections.filter(function (detection) {
        return !PEOPLE_CLASSES.includes(detection.name);
      });
    }
    detections = steadyDetections(detections);

    drawBoxes(detections);
    showResult(detections);

    // Wait for the browser to draw the next frame before detecting again.
    // This keeps the page smooth and responsive.
    await new Promise(requestAnimationFrame);
  }
}

// Keeps boxes steady. The model sometimes misses an object for one frame,
// which would make its box flicker off and on. So an object from earlier
// frames stays on screen for HOLD_TIME, unless the new frame already found
// the same object (same name, overlapping box).
function steadyDetections(detections) {
  const now = performance.now();

  const stillShown = shownObjects.filter(function (old) {
    const recentlySeen = now - old.lastSeen < HOLD_TIME;
    const foundAgain = detections.some(function (detection) {
      return detection.name === old.name && overlap(detection, old) > 0.3;
    });
    return recentlySeen && !foundAgain;
  });

  detections.forEach(function (detection) {
    detection.lastSeen = now;
  });

  shownObjects = detections.concat(stillShown);
  return shownObjects.slice().sort(function (a, b) {
    return b.score - a.score; // most confident first
  });
}

// ---------- 9. Draw a box and a label around every detected object ----------
function drawBoxes(detections) {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);

  // Scale the line and text with the video size so they stay readable.
  const lineWidth = Math.max(2, overlay.width / 200);
  const fontSize = Math.max(14, overlay.width / 32);
  overlayContext.lineWidth = lineWidth;
  overlayContext.font = "600 " + fontSize + "px Segoe UI, sans-serif";
  overlayContext.textBaseline = "top";

  detections.forEach(function (detection) {
    const text = detection.name + " " + Math.round(detection.score * 100) + "%";

    // The box
    overlayContext.strokeStyle = "#38bdf8";
    overlayContext.strokeRect(detection.x, detection.y, detection.width, detection.height);

    // A filled strip behind the text, then the text itself.
    // Keep it inside the picture when the box touches the top edge.
    const textWidth = overlayContext.measureText(text).width;
    const textY = Math.max(0, detection.y - fontSize - 6);
    overlayContext.fillStyle = "#38bdf8";
    overlayContext.fillRect(detection.x, textY, textWidth + 10, fontSize + 6);
    overlayContext.fillStyle = "#0f172a";
    overlayContext.fillText(text, detection.x + 5, textY + 3);
  });
}

// ---------- 10. Show the results in the side panel ----------
function showResult(detections) {
  if (detections.length === 0) {
    labelEl.textContent = "No object detected";
    confidenceEl.textContent = "—";
    confidenceBar.style.width = "0%";
    otherObjectsEl.innerHTML = '<li class="empty">Nothing else in view.</li>';
    return;
  }

  // The first detection is the most confident one (sorted in detectObjects).
  const best = detections[0];
  const percent = (best.score * 100).toFixed(1) + "%";
  labelEl.textContent = best.name;
  confidenceEl.textContent = percent;
  confidenceBar.style.width = percent;

  // Every other detected object goes in the list below.
  otherObjectsEl.innerHTML = "";
  if (detections.length === 1) {
    otherObjectsEl.innerHTML = '<li class="empty">Nothing else in view.</li>';
  }
  detections.slice(1).forEach(function (detection) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const score = document.createElement("span");
    name.textContent = detection.name;
    score.textContent = (detection.score * 100).toFixed(1) + "%";
    item.appendChild(name);
    item.appendChild(score);
    otherObjectsEl.appendChild(item);
  });
}

function resetResult() {
  labelEl.textContent = "—";
  confidenceEl.textContent = "—";
  confidenceBar.style.width = "0%";
  otherObjectsEl.innerHTML = '<li class="empty">Start the camera to see detections.</li>';
}

// ---------- 11. Wire up the buttons and start loading ----------
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
cameraSelect.addEventListener("change", switchCamera);

loadModel();
