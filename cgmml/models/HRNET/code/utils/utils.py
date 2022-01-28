import cgmml.models.HRNET.code.models.pose_hrnet  # noqa
import math
import cv2
import logging
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from cgmml.models.HRNET.code.config import cfg
from cgmml.models.HRNET.code.config.constants import COCO_INSTANCE_CATEGORY_NAMES, NUM_KPTS, SKELETON, CocoColors
from cgmml.models.HRNET.code.utils.post_processing import get_final_preds
from cgmml.models.HRNET.code.utils.transforms import get_affine_transform
from cgmml.models.HRNET.code.config.constants import FACE_TYPE_ONE, FACE_TYPE_TWO

logging.basicConfig(level=logging.INFO, filename='pose_prediction.log',
                    format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', filemode='w')


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return [], 0

    filtered_index = [pred_score.index(x) for x in pred_score if x > threshold]
    pred_boxes = [pred_boxes[idx] for idx in filtered_index]
    pred_score = [pred_score[idx] for idx in filtered_index]
    pred_classes = [pred_classes[idx] for idx in filtered_index]

    person_boxes, person_scores = [], []
    for box, score, class_ in zip(pred_boxes, pred_score, pred_classes):
        if class_ == 'person':
            person_boxes.append(box)
            person_scores.append(score)

    return person_boxes, person_scores


def rot(keypoints, orientation, height, width):
    """
    Rotate a point counterclockwise,or clockwise.
    """
    rotated_keypoints = list()
    for i in range(0, NUM_KPTS):
        if orientation == 'ROTATE_90_CLOCKWISE':
            rot_x, rot_y = width - keypoints[i][1], keypoints[i][0]
        elif orientation == 'ROTATE_90_COUNTERCLOCKWISE':
            rot_x, rot_y = keypoints[i][1], height - keypoints[i][0]
        rotated_keypoints.append([rot_x, rot_y])
    return rotated_keypoints


def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def perpendicular_distance(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    dist = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

    return dist


def orient_axis(p):
    q = (-p[1], p[0])
    return q


def reorient_to_original_axis(p):
    q = [-p[1], p[0]]
    return q


def get_perpendicular_points(p1, p2, n):
    x1, y1 = p1[1], p1[0]
    x2, y2 = p2[1], p2[0]

    vx = x2 - x1
    vy = y2 - y1
    len = math.sqrt(vx * vx + vy * vy)
    ux = -vy / len
    uy = vx / len

    x3 = x1 + n * ux
    y3 = y1 + n * uy
    x4 = x2 + n * ux
    y4 = y2 + n * uy

    p3, p4 = [y3, x3], [y4, x4]
    return p3, p4


def contour_using_eye_nose_shoulder(keypoints):
    left_shoulder, right_shoulder = keypoints[FACE_TYPE_TWO[5]], keypoints[FACE_TYPE_TWO[6]]
    middle_point = [min(left_shoulder[0], right_shoulder[0]), (left_shoulder[1] + right_shoulder[1]) / 2]

    contour = [keypoints[idx] for idx in FACE_TYPE_ONE]
    contour.append(middle_point)

    logging.info("%s %s", "left_shoulder ", left_shoulder)
    logging.info("%s %s", "right_shoulder ", right_shoulder)
    logging.info("%s %s", "middle_point ", middle_point)
    logging.info("%s %s", "contour using prepare_contour_using_shoulder ", contour)

    return contour


def contour_using_eye_nose(keypoints):
    '''                                 y
                                        |
                                        |
                                        | index 0
                                        |
                                        |
                                        |
                                        |
                                        |
    x ___________________________________
        index 1
    '''

    # nose, left_eye, right_eye = keypoints['nose'], keypoints['left_eye'], keypoints['right_eye']  #error
    nose, left_eye, right_eye = keypoints[0], keypoints[1], keypoints[2]

    middle_eye_point = [min(left_eye[0], right_eye[0]), (left_eye[1] + right_eye[1]) / 2]
    distance_bw_eyes = (right_eye[1] - left_eye[1]) / 2
    # distance_bw_eye_nose_using_brute = middle_eye_point[0] - nose[0]
    distance_bw_eye_nose = perpendicular_distance(left_eye, right_eye, nose)

    left_eye = [left_eye[0], max(0, left_eye[1] - distance_bw_eyes)]
    right_eye = [right_eye[0], right_eye[1] + distance_bw_eyes]

    p1, p2 = get_perpendicular_points(left_eye, right_eye, -1.5 * distance_bw_eye_nose)
    p3, p4 = get_perpendicular_points(left_eye, right_eye, 3 * distance_bw_eye_nose)

    contour = [
        left_eye, right_eye,
        p1, p2,
        p3, p4
    ]

    logging.info('----------------------------------------------')
    logging.info("%s %s %s %s %s %s", "left eye ", left_eye, " right eye ", right_eye, " nose ", nose)
    logging.info("%s %s", "middle_eye_point ", middle_eye_point)
    logging.info("%s %s", "distance_bw_eyes ", distance_bw_eyes)
    # logging.info("%s %s", "distance_bw_eye_nose_primitive using brute ", distance_bw_eye_nose_using_brute)
    logging.info("%s %s", "distance_bw_eye_nose using perpendicular function ", distance_bw_eye_nose)
    logging.info("%s %s %s %s %s %s %s %s", "p1 ", p1, " p2 ", p2, " p3 ", p3, " p4 ", p4)

    logging.info("%s %s", "contour_using_eye_nose ", contour)

    logging.info('----------------------------------------------')
    return contour


def draw_face_blur_using_pose_basic(keypoints, img):
    """perform the face blur using the contour drawn from basic keypoints.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    contour = [keypoints[idx] for idx in FACE_TYPE_ONE]
    # contour = np.array(np.ceil(contour), dtype=int)
    contour = np.array(contour, dtype=int)
    hull = cv2.convexHull(contour)
    logging.info("%s %s", "contour ", contour)
    logging.info("%s %s", "hull ", hull)
    cv2.fillPoly(img, pts=[hull], color=(234, 237, 237))


def draw_face_blur_using_pose_advance(keypoints, img):
    """perform the face blur using the contour drawn from keypoints using advanced method.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)

    # contour = contour_using_shoulder(keypoints)
    contour = contour_using_eye_nose(keypoints)
    # logging.info("%s %s", "contour ", contour)

    contour = np.array(np.ceil(contour), dtype=int)
    hull = cv2.convexHull(contour)

    # ellipse = cv2.fitEllipse(contour)
    # logging.info("%s %s", "ellipse ", ellipse)
    logging.info("%s %s", "hull ", hull)

    cv2.fillPoly(img, pts=[hull], color=(234, 237, 237))


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, score = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))
        return preds, score


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def calculate_pose_score(pose_score):
    return np.mean(pose_score)
