import cv2
import dlib
from facenet_pytorch import MTCNN
import face_alignment
import numpy as np
from PIL import Image
import scipy
import scipy.ndimage
from importlib_resources import files
from .matlab_cp2tform import get_similarity_transform_for_cv2
from .align_trans import get_affine_transform_matrix


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


def detect_landmark_facenet(image):
    if not 'mtcnn' in globals():
        global mtcnn
        mtcnn = MTCNN()
        
    _, _, landmarks = mtcnn.detect(image, landmarks=True)
    landmarks = np.array(landmarks).astype(np.int32)[0]
    return landmarks


def detect_landmark_3dlm(image):
    if not 'fa' in globals():
        global fa
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    
    image = np.array(image)
    preds = fa.get_landmarks(image)

    lms = np.array(preds[0])
    lm_nose          = lms[30]
    lm_eye_left      = lms[36 : 42, :2]
    lm_eye_right     = lms[42 : 48, :2]
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    mouth_left   = lms[48]
    mouth_right  = lms[54]

    landmarks = np.array([eye_left,eye_right,lm_nose, mouth_left,mouth_right]).astype(np.int32)
    return landmarks


def detect_landmark_dlib(image):
    if not 'detector' in globals():
        global detector
        detector = dlib.get_frontal_face_detector()
    if not 'predictor' in globals():
        filename = 'shape_predictor_68_face_landmarks.dat'
        source = files('face').joinpath(filename)
        path = str(source)
        global predictor
        predictor = dlib.shape_predictor(path)
    
    image = np.array(image)
    rect = detector(image, 1)[0]
    shape = predictor(image, rect)
    preds = [(shape.part(j).x, shape.part(j).y) for j in range(68)]

    lm = np.array(preds)
    lm_nose          = lm[30]
    lm_eye_left      = lm[36 : 42, :2]
    lm_eye_right     = lm[42 : 48, :2]
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    mouth_left   = lm[48]
    mouth_right  = lm[54]

    landmarks = np.array([eye_left,eye_right,lm_nose, mouth_left,mouth_right]).astype(np.int32)
    return landmarks


def align_cv2(src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type='smilarity'):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    ref_pts = np.float32(reference_pts)
    ref_pts = (ref_pts - 112/2)*0.72 + 112/2
    ref_pts *= crop_size[0]/112.
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm, tfm_inv = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]), borderMode=cv2.BORDER_REFLECT)

    return face_img, tfm_inv


def align_ffhq(img, face_landmarks, output_size=256, transform_size=1024, enable_padding=True, img_dict=None):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    # Calculate auxiliary vectors.
    eye_left     = face_landmarks[0]
    eye_right    = face_landmarks[1]
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = face_landmarks[3]
    mouth_right  = face_landmarks[4]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad_ = quad.copy()
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    print(shrink)
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img, quad_, img_dict


def align_new(img, face_landmarks, output_size=256):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    
    # Calculate auxiliary vectors.
    eye_left     = face_landmarks[0]
    eye_right    = face_landmarks[1]
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = face_landmarks[3]
    mouth_right  = face_landmarks[4]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    src_pts = quad
    ref_pts = np.array(((0, 0), (0, output_size), (output_size, output_size), (output_size, 0)))
    tfm, tfm_inv = get_similarity_transform_for_cv2(src_pts, ref_pts)
    face_img = cv2.warpAffine(np.array(img), tfm, (output_size, output_size), borderMode=cv2.BORDER_REFLECT)

    return face_img, tfm_inv
