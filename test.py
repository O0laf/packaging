import numpy as np
import cv2
from PIL import Image
import os
import glob
import argparse
import face as f


parser = argparse.ArgumentParser()

parser.add_argument('--align_type', default="ffhq", help="cv2 or ffhq or new")
parser.add_argument('--detect_type', default="dlib", help="facenet or 3dlm or dlib")

parser.add_argument('--size', default=1024, help="size of face")
parser.add_argument('--image_root', default="image_input")
parser.add_argument('--result_dir', default="image_output")

args = parser.parse_args()

face_size = np.array([args.size, args.size])
ref5points = f.get_reference_facial_points(default_square=True)
img_path_list = glob.glob(f"{args.image_root}/*.*")
img_dict = {}

# Mask
half = args.size // 2
mask_grad = np.zeros(face_size, dtype=float)

for i in range(args.size):
    for j in range(args.size):
        dist = np.sqrt((i-half)**2 + (j-half)**2)/half
        dist = np.minimum(dist, 1)
        mask_grad[i, j] = (1-dist)
mask_grad = cv2.dilate(mask_grad, None, iterations=30)
mask_grad = np.expand_dims(mask_grad, 2).repeat(3, axis=2)
mask_grad = Image.fromarray((mask_grad*255).astype(np.uint8))

for img_path in img_path_list:

    fname = os.path.split(img_path)[1][:-4] 
    save_dir = f"{args.result_dir}/{args.detect_type}_{args.align_type}_{args.size}/{fname}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n >>> processing {fname}\n")

    ##################################
    ### 1. load image 
    ##################################

    img = Image.open(img_path)

    img_dict["original"] = img

    ##################################
    ### 2. detect landmarks
    ##################################

    img_w, img_h = img.width, img.height
    img_ = img.copy().resize((img_w//2, img_h//2))
    img_ = np.array(img_)

    try:
        lms_facenet = f.detect_landmark_facenet(img)
        lms_3dlm = f.detect_landmark_3dlm(img_)
        lms = f.detect_landmark_dlib(img_)
            
    except Exception as e:
        print(e)
        print("fail to detect face")
        continue
    
    if lms is None: 
        continue

    lms *= 2
    lms_facenet *= 2
    lms_3dlm *= 2
    
    ##################################
    ### 3. draw landmarks
    ##################################

    img_ = np.array(img)

    for (x, y) in lms:
        cv2.circle(img_,(int(x), int(y)), 2, (0,0,255), 2)
    for (x, y) in lms_3dlm:
        cv2.circle(img_,(int(x), int(y)), 2, (0,255,0), 2)
    for (x, y) in lms_facenet:
        cv2.circle(img_,(int(x), int(y)), 2, (255,0,0), 2)

    img_dict["landmark"] = Image.fromarray(img_.astype(np.uint8))
    
    ##################################
    ### 4. align face
    ##################################
    
    if args.align_type == "cv2":
        face, trans_inv = f.align_cv2(np.array(img), lms, ref5points, crop_size=face_size)
        face = Image.fromarray(face.astype(np.uint8))

    if args.align_type == "ffhq":
        face, quad, img_dict = f.align_ffhq(img, lms, output_size=args.size, transform_size=args.size*4, img_dict=img_dict)

    if args.align_type == "new":
        face, trans_inv = f.align_new(img, lms, output_size=args.size)
        face = Image.fromarray(face.astype(np.uint8))

    img_dict["aligned"] = face

    # Save images
    for key, value in img_dict.items():
        value.save(f"{save_dir}/{key}.jpg")
