import glob

import numpy as np

import face_alignment_simple
from skimage import io
import cv2


# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold" : 0.8
}

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment_simple.FaceAlignment(face_alignment_simple.LandmarksType._2D, device='cpu', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

for image_file in glob.glob('test_data/*.png'):

    input_img = io.imread(image_file)

    preds = fa.get_landmarks(input_img)[-1]

    scale = 4

    output_img = cv2.resize(input_img, (0, 0), fx=scale, fy=scale)
    output_img = np.ascontiguousarray(output_img[..., ::-1])

    preds *= scale

    for i in range(0, preds.shape[0] - 1):
        p0 = tuple(preds[i].astype(int))
        # p1 = tuple(preds[i + 1].astype(int))
        # cv2.line(output_img, p0, p1, (0, 255, 0))
        cv2.circle(output_img, p0, 2, (0, 255, 0))
        text_params = {
            'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
            'fontScale': 0.5,
            'thickness': 1,
            'color': (0, 255, 0)
        }
        cv2.putText(output_img, f'{i}', p0, **text_params)

    cv2.imshow('keypoints', output_img)
    cv2.waitKey(0)
