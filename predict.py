import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("modeltrain.h5")   # <-- rename if different

# Load an image you want to test
img_path = "test.jpg"   # <-- put any face image here
image = cv2.imread(img_path)

# Preprocess same as training
image_resized = cv2.resize(image, (64, 64))   # YOUR training size may differ
image_normalized = image_resized / 255.0
image_input = np.expand_dims(image_normalized, axis=0)

# Predict
prediction = model.predict(image_input)
label = "Real Face" if prediction[0][0] > 0.5 else "Spoof / Fake Face"

print("Prediction:", label)
