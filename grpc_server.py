#!/usr/bin/env python3
import time
import py_protos.detection_v2_pb2 as grpc_msg
import py_protos.detection_v2_pb2_grpc as grpc
import grpc as grpclib
import cv2
from concurrent import futures
import numpy as np
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess_custom, upscale_masks
import torch
import torch.backends.cudnn as cudnn
import argparse
from utils.functions import SavePath

_NUMPY_TYPES_MAP = {
    "uint8": np.uint8,
    "float32": np.float32,
    'bool': np.bool
}

from data.config import COLORS, cfg, set_cfg, set_dataset

def img_msg_from_array(img):
    msg = grpc_msg.Image()
    msg.height, msg.width, msg.channel = img.shape
    msg.data = bytes(img.reshape(-1))
    msg.type = img.dtype.name
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

class InstanceDetectionServiceServer(grpc.InstanceDetectionServiceServicer):
    def __init__(self, args):
        self._model = Yolact()
        print('Loading the model at: {}'.format(args.model_weight) )
        self._model.load_weights(args.model_weight)
        self._model.eval()
        self._model.detect.use_fast_nms = args.use_fast_nms
        self._top_k = args.top_k
        self._score_threshold = args.score_threshold
        self._visualize = args.visualize
        self._full_mask = args.full_mask
        print("Initializing ..")
        self._detect(torch.zeros(640, 480, 3) )  # run detection once to initialize
        print("Done")

    # Image input type: BGR
    def DetectInstances(self, image_msg, context):
        class_idxs, scores, boxes_points, masks, draw_img = self._detect(img_msg_to_array(image_msg))
        # TODO: add to viz thread, now test by drawing here!
        if self._visualize:
            self._draw_result(draw_img, class_idxs, scores, boxes_points)
        msg = grpc_msg.InstanceDetections()
        preds = []
        boxes = center_size_boxes(boxes_points)

        for class_idx, score, box, mask in zip(class_idxs, scores, boxes, masks):
            p = grpc_msg.InstanceDetection()
            p.confidence = score
            p.label_id = class_idx
            p.mask = img_msg_from_array(mask.astype(np.bool))
            p.box.x, p.box.y, p.box.w, p.box.h = box[0], box[1], box[2], box[3]
            p.mask_type = int(self._full_mask)
            preds.append(p)

        msg.predictions.extend(preds)
        return msg


    def _detect(self, image):
        frame = torch.Tensor(image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))  # GPU SWAP
        t = postprocess_custom(self._model(batch),
                               image.shape[1],
                               image.shape[0],
                               get_full_mask= self._full_mask,
                               score_threshold= self._score_threshold)
        torch.cuda.synchronize()

        if self._visualize:
            masks = t[3][:self._top_k]  # mask_gpu We'll need this later
            classes, scores, boxes = [x[:self._top_k].cpu().numpy() for x in t[:3]]

            img_gpu = image / 255.0
            h, w, _ = image.shape
            if classes.shape[0] == 0:
                img_numpy = (img_gpu * 255).byte().cpu().numpy()
                return classes, scores, boxes, masks, img_numpy

            if not self._full_mask:
                full_masks = upscale_masks(masks, w, h, boxes)
            else:
                full_masks = masks

            # Drawing mask in GPU
            for j in reversed(range(min(self._top_k, classes.shape[0]))):
                if scores[j] >= self._score_threshold:
                    color = get_color(j, classes)

                    mask = full_masks[j, :, :, None]
                    mask_color = mask @ (torch.Tensor(color).view(1, 3) / 255.0)
                    mask_alpha = 0.45

                    # Alpha only the region of the image that contains the mask
                    img_gpu = img_gpu * (1 - mask) \
                              + img_gpu * mask * (1 - mask_alpha) + mask_color * mask_alpha

            img_numpy = (img_gpu * 255).byte().cpu().numpy()
            masks_cpu = t[3][:self._top_k].cpu().numpy()
            return classes, scores, boxes, masks_cpu, img_numpy
        else:
            classes, scores, boxes, masks = [x[:self._top_k].cpu().numpy() for x in t]
            return classes, scores, boxes, masks, None

    # draw result on cpu, box recieved as corners
    def _draw_result(self, predrawn_frame, classes, scores, boxes):
        for j in reversed(range(min(self._top_k, classes.shape[0]))):
            score = scores[j]

            if scores[j] >= self._score_threshold:
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Yolact GRPC Inference server'
    )
    parser.add_argument('--model_weight', required=True,help='Path to .pth weight')
    parser.add_argument('--use_fast_nms', default=False, help='Use fast (coarse) NMS')
    parser.add_argument('--score_threshold', default=0.0, help='Minimum Score threshold')
    parser.add_argument('--top_k', default=100, help='Total top instances to return')
    parser.add_argument('--visualize', default=True, help='Whether to run visualization window')
    parser.add_argument('--full_mask', default=False, help='Whether to get full mask, otherwise cropped')
    parser.add_argument('--cpu', action='store_true', help='Whether to use cpu for calc')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    args = parser.parse_args()
    if args.config is None:
        model_path = SavePath.from_str(args.model_weight)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    server = grpclib.server(futures.ThreadPoolExecutor(max_workers=1))
    with torch.no_grad():
        # pre configure
        if args.cpu:
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        grpc.add_InstanceDetectionServiceServicer_to_server(
            InstanceDetectionServiceServer(args),
            server)
        server.add_insecure_port('[::]:50051')
        server.start()
        print('Starting server at localhost:50051')
        try:
            while True:
                time.sleep(100)
        except KeyboardInterrupt:
            server.stop(0)

if __name__ == '__main__':
    main()
