"""
Inference handler for SageMaker TensorFlow Serving endpoint.
Handles image preprocessing and prediction for the deployed scene classification model.
"""

import os, json, base64
import numpy as np
import tensorflow as tf

def _find_savedmodel(root):
    v1 = os.path.join(root, "1")
    if os.path.exists(os.path.join(v1, "saved_model.pb")):
        return v1
    if os.path.exists(os.path.join(root, "saved_model.pb")):
        return root
    for r, _, f in os.walk(root):
        if "saved_model.pb" in f:
            return r
    raise FileNotFoundError("saved_model.pb not found under " + root)

def model_fn(model_dir):
    sm_path = _find_savedmodel(model_dir)
    return tf.saved_model.load(sm_path)

def input_fn(request_body, request_content_type):
    # Return raw bytes; decode later in predict_fn so we don't need PIL
    if request_content_type == "application/x-image":
        return request_body  # bytes
    # assume JSON with base64
    payload = json.loads(request_body)
    return base64.b64decode(payload["b64"])

def _preprocess(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)  # handles JPG/PNG
    img = tf.image.resize(img, [150, 150])
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)  # [1,H,W,3]

def predict_fn(img_bytes, model):
    arr = _preprocess(img_bytes)
    out = list(model.signatures["serving_default"](arr).values())[0].numpy()[0].tolist()
    return out

def output_fn(prediction, accept):
    return json.dumps({"probs": prediction}), "application/json"