import py_protos.detection_v2_pb2 as grpc_msg
import cv2
import numpy as np
from data.config import COLORS, cfg

_NUMPY_TYPES_MAP = {
    "uint8": np.uint8,
    "float32": np.float32,
    'bool': np.bool
}

def img_msg_from_array(img, rgb=False):
    msg = grpc_msg.Image()

    if len(img.shape) == 2:
        img_out = img.reshape(*img.shape, 1)
    else:
        img_out = img

    assert len(img_out.shape) == 3

    msg.height, msg.width, msg.channel = img_out.shape
    msg.data = bytes(img_out.reshape(-1)) # as raw boolean still encode to uint8t, but with only 0, 1 values
    msg.type = img_out.dtype.name
    msg.rgb = rgb
    return msg

def img_msg_to_array(msg: grpc_msg.Image, swapRB=False):
    np_data = np.frombuffer(msg.data, dtype=_NUMPY_TYPES_MAP[msg.type])
    img = np_data.reshape((msg.height, msg.width, msg.channel))
    if swapRB:
        return img[:,:,::-1]
    else:
        return img

def get_color(j, classes):
    color = COLORS[(classes[j] * 5) % len(COLORS)]
    color = (color[2], color[1], color[0])
    return color

def center_size_boxes(boxes):
    centers = (boxes[:, 2:] + boxes[:, :2])/2
    wh = (boxes[:, 2:] - boxes[:, :2])
    return np.concatenate((centers, wh), axis=1)


# draw result on cpu, box recieved as corners
def draw_result(frame, classes, scores, boxes, score_threshold=0):
    predrawn_frame = np.copy(frame)
    for j in reversed(range(classes.shape[0])):
        score = scores[j]

        if scores[j] >= score_threshold:
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j, classes)

            # if args.display_bboxes:
            cv2.rectangle(predrawn_frame, (x1, y1), (x2, y2), color, 1)


            _class = cfg.dataset.class_names[classes[j]]
            text_str = '%s: %.2f' % (_class, score)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

            text_pt = (x1, y1 - 3)
            text_color = [255, 255, 255]

            cv2.rectangle(predrawn_frame, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            cv2.putText(predrawn_frame, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)

    return predrawn_frame