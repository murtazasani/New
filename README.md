import cv2
import numpy as np

def alpha_composite(image1, image2, alpha):
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

def calculate_mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

vidcap = cv2.VideoCapture('sample.mp4')
count = 0
frame_gap = 25
previous_frame = None
threshold_mse = 100.0
output_image = None

while True:
    success, image = vidcap.read()

    if not success:
        break

    if count % frame_gap == 0:
        if previous_frame is not None and calculate_mse(image, previous_frame) > threshold_mse:
            if output_image is not None:
                cv2.imwrite(f"output_frame_{count // frame_gap}.jpg", output_image)

            output_image = previous_frame.copy()

    previous_frame = image

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

    count += 1

vidcap.release()

# Save the last merged composite image
if output_image is not None:
    cv2.imwrite(f"images_generated/output_frame_{count // frame_gap}.jpg", output_image)

cv2.destroyAllWindows()
