import cv2
import numpy as np
import face_recognition
from skimage import exposure
import os

def enhance_image(image_path, zoom_factor=1.2):
    """Enhance image for better face detection and matching."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Cannot load image: " + image_path)

    # 1. Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Face alignment using eye landmarks
    try:
        face_locations = face_recognition.face_locations(rgb_image)
        if face_locations:
            landmarks = face_recognition.face_landmarks(rgb_image)
            if landmarks and 'left_eye' in landmarks[0] and 'right_eye' in landmarks[0]:
                left_eye = np.mean(landmarks[0]['left_eye'], axis=0)
                right_eye = np.mean(landmarks[0]['right_eye'], axis=0)

                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dy, dx))

                center = tuple(np.array(rgb_image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                rgb_image = cv2.warpAffine(rgb_image, rot_mat, rgb_image.shape[1::-1], flags=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"[WARNING] Face alignment skipped: {e}")

    # 3. Histogram equalization for contrast enhancement
    for i in range(3):
        channel = rgb_image[:, :, i]
        eq_channel = exposure.equalize_hist(channel)
        rgb_image[:, :, i] = np.clip(eq_channel * 255, 0, 255)
    rgb_image = rgb_image.astype(np.uint8)

    # 4. Image sharpening
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    rgb_image = cv2.filter2D(rgb_image, -1, sharpening_kernel)

    # 5. Optional center zoom
    if zoom_factor != 1.0:
        h, w = rgb_image.shape[:2]
        y1 = int(h // 2 - h // (2 * zoom_factor))
        y2 = int(h // 2 + h // (2 * zoom_factor))
        x1 = int(w // 2 - w // (2 * zoom_factor))
        x2 = int(w // 2 + w // (2 * zoom_factor))
        zoomed = rgb_image[y1:y2, x1:x2]
        rgb_image = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_CUBIC)

    # Save enhanced version in temp folder
    os.makedirs("static/temp_enhanced", exist_ok=True)
    filename = os.path.basename(image_path)
    temp_path = os.path.join("static", "temp_enhanced", filename)
    cv2.imwrite(temp_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    return temp_path
