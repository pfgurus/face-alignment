import glob
import os
import argparse
import numpy as np

import face_alignment_simple
from skimage import io
import cv2


def find_landmarks(args):
    # Optionally set detector and some additional detector parameters
    face_detector = 'sfd'
    face_detector_kwargs = {
        "filter_threshold": 0.8
    }

    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment_simple.FaceAlignment(face_alignment_simple.LandmarksType._2D, device='cpu', flip_input=True,
                                      face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
    os.makedirs(args.output_dir, exist_ok=True)
    for input_file in glob.glob(f'{args.input_dir}/*.png'):

        input_image = io.imread(input_file)

        preds = fa.get_landmarks(input_image)[-1]

        scale = 4

        output_image = cv2.resize(input_image, (0, 0), fx=scale, fy=scale)
        output_image = np.ascontiguousarray(output_image[..., ::-1])

        preds *= scale

        for i in range(0, preds.shape[0] - 1):
            p0 = tuple(preds[i].astype(int))
            # p1 = tuple(preds[i + 1].astype(int))
            # cv2.line(output_image, p0, p1, (0, 255, 0))
            cv2.circle(output_image, p0, 2, (0, 255, 0))
            text_params = {
                'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
                'fontScale': 0.5,
                'thickness': 1,
                'color': (0, 255, 0)
            }
            cv2.putText(output_image, f'{i}', p0, **text_params)

        output_file = os.path.splitext(os.path.basename(input_file))[0] + '.jpg'
        cv2.imwrite(os.path.join(args.output_dir, output_file), output_image)
        # cv2.imshow('keypoints', output_image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='test_data', help="Input directory with images (PNG)")
    parser.add_argument("--output_dir", default='output', help="Output directory")
    parser.add_argument("--output_scale", default=3, help="Output image scale")
    args = parser.parse_args()

    find_landmarks(args)