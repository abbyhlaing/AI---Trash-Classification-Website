from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import io
import base64
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

app = Flask(__name__)

# ✅ Load model ONCE (important for performance)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(
    "recycle_model_full.pt",
    map_location=device,
    weights_only=False   # ✅ IMPORTANT
)
model.to(device)
model.eval()

yolo = YOLO("yolov8n.pt")

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ✅ Prediction function (reuse everywhere)
def model_predict(img_pil):
    with torch.inference_mode():
        img = transform(img_pil).unsqueeze(0).to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)

        entropy = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        max_entropy = torch.log(torch.tensor(float(len(class_names))).to(device)).item()

        if entropy > 0.7 * max_entropy:
            return "Not Trash / Unknown", 0.0

        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs.max().item() * 100

    return pred_class, confidence


# ================= ROUTES =================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/team")
def team():
    return render_template("team.html")


@app.route("/user-guide")
def guide():
    return render_template("guide.html")


@app.route("/live-cam")
def livecam():
    return render_template("livecam.html")

@app.route("/testing")
def testing():
    return render_template("big_testing.html")


# ✅ OLD IMAGE UPLOAD (kept)
@app.route("/", methods=["POST"])
def predict_upload():
    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template("index.html", error="No image selected")

    img_pil = Image.open(file).convert("RGB")

    pred_class, confidence = model_predict(img_pil)

    # Convert image to base64 (for preview)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template(
        "index.html",
        prediction=pred_class,
        confidence=f"{confidence:.2f}",
        img_data=img_str
    )


# ✅ 🔥 NEW REAL-TIME API (THIS IS THE KEY)
@app.route("/predict", methods=["POST"])
def predict_realtime():
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No image"}), 400

    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)

    # YOLO detection
    results = yolo(img_np)[0]

    boxes_data = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # if cls_id == 0:
        #     continue
    
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 

        # 🔥 Crop object
        cropped = img.crop((x1, y1, x2, y2))

        # 🔥 Classify with YOUR model
        label, confidence = model_predict(cropped)

        # Optional: skip unknown
        if label == "Not Trash / Unknown":
            continue

        boxes_data.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "prediction": label,
            "confidence": round(confidence, 2)
        })
    
    # print("results ==> ", results)
    print("Detected boxes:", boxes_data)

    return jsonify({"boxes": boxes_data})


# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5300, debug=True)