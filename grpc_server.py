#!/usr/bin/env python3
import time
import py_protos.detection_v2_pb2_grpc as grpc
import grpc as grpclib
from concurrent import futures
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
import torch
import torch.backends.cudnn as cudnn
import argparse
from utils.functions import SavePath
import os
from py_protos.common_utils import *
from data.config import set_cfg


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
        self._full_mask = not args.cropped_mask
        print("Initializing ..", end='')
        _initimg = cv2.imread(os.path.join(os.getcwd(), 'data', 'grpc_init.png'))

        if _initimg is None:
            print('Fail to load initial image')
            _initimg = np.zeros((640, 480, 3))

        self._detect(_initimg)  # run detection once to initialize

        print("Done")

    # Image input type: BGR
    def DetectInstances(self, image_msg, context):
        with torch.no_grad():

            class_idxs, scores, boxes, masks, draw_img = self._detect(img_msg_to_array(image_msg))

            if self._visualize:
                draw_img = draw_result(draw_img, class_idxs, scores, boxes)

        msg = grpc_msg.InstanceDetections()
        preds = []
        boxes = boxes.astype(np.int)

        for class_idx, score, box, mask in zip(class_idxs, scores, boxes, masks):
            p = grpc_msg.InstanceDetection()
            p.confidence = score
            p.label_id = class_idx
            p.box.tlx, p.box.tly, p.box.brx, p.box.bry = box[0], box[1], box[2], box[3]

            if self._full_mask:
                p.mask.CopyFrom(img_msg_from_array(mask.astype(np.bool)))
            else:
                p.mask.CopyFrom(img_msg_from_array(mask[box[1]:box[3]+1, box[0]:box[2]+1].astype(np.bool)))

            print(mask.shape)
            cv2.imwrite('debug_server{}.jpg'.format(class_idx),
                        mask[box[1]:box[3] + 1, box[0]:box[2] + 1].astype(np.uint8) * 255)

            p.mask_type = int(self._full_mask)
            preds.append(p)

        msg.predictions.extend(preds)
        return msg


    def _detect(self, image):
        frame = torch.Tensor(image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))  # GPU SWAP
        t = postprocess(self._model(batch),
                               image.shape[1],
                               image.shape[0],
                               score_threshold= self._score_threshold)
        torch.cuda.synchronize()

        if self._visualize:
            masks = t[3][:self._top_k]  # mask_gpu We'll need this later
            classes, scores, boxes = [x[:self._top_k].cpu().numpy() for x in t[:3]]

            img_gpu = frame / 255.0
            h, w, _ = frame.shape
            if classes.shape[0] == 0:
                img_numpy = (img_gpu * 255).byte().cpu().numpy()
                return classes, scores, boxes, masks, img_numpy

            # Drawing mask in GPU
            for j in reversed(range(min(self._top_k, classes.shape[0]))):
                if scores[j] >= self._score_threshold:
                    color = get_color(j, classes)

                    mask = masks[j, :, :, None]
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Yolact GRPC Inference server'
    )
    parser.add_argument('--model_weight', required=True,help='Path to .pth weight')
    parser.add_argument('--use_fast_nms', default=False, help='Use fast (coarse) NMS')
    parser.add_argument('--score_threshold', default=0.3, help='Minimum Score threshold')
    parser.add_argument('--top_k', default=100, help='Total top instances to return')
    parser.add_argument('--visualize', default=True, help='Whether to run visualization window')
    parser.add_argument('--cropped_mask', type=str2bool, nargs="?", const=True, help='Whether to get full mask, otherwise cropped')
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
