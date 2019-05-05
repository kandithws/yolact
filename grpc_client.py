import cv2
from sys import argv
from py_protos.common_utils import img_msg_from_array, img_msg_to_array
import py_protos.detection_v2_pb2_grpc as grpc
import grpc as grpclib
from py_protos.common_utils import draw_result
import numpy as np

def main():
    with grpclib.insecure_channel('0.0.0.0:50051') as channel:
        stub = grpc.InstanceDetectionServiceStub(channel)
        in_img = cv2.imread(argv[1])
        msg = img_msg_from_array(in_img)
        response = stub.DetectInstances(msg)

    classes = []
    scores = []
    bbs = []
    masks = []
    for pred in response.predictions:
        classes.append(pred.label_id)
        bbs.append( (pred.box.tlx, pred.box.tly, pred.box.brx, pred.box.bry) )
        scores.append(pred.confidence)
        if pred.mask_type == 0:
            print("I AM HERE!")
            cropped_mask = img_msg_to_array(pred.mask)
            full_mask = np.zeros((*in_img.shape[:2],1), dtype=np.bool)
            print(full_mask[pred.box.tly:pred.box.bry+1, pred.box.tlx:pred.box.brx+1].shape)
            print(cropped_mask.shape)
            full_mask[pred.box.tly:pred.box.bry+1, pred.box.tlx:pred.box.brx+1] = cropped_mask
            masks.append(full_mask.astype(np.uint8) * 255)
        else: # full_mask
            masks.append(img_msg_to_array(pred.mask).astype(np.uint8) * 255)

    out = draw_result(in_img, np.array(classes), np.array(scores), np.array(bbs) )
    cv2.imwrite('out_client.jpg', out)

    for i, mask in enumerate(masks):
        cv2.imwrite('client_mask{}.jpg'.format(i), mask)

    print("--------DONE----------")


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: client.py [input image]")
        exit(0)
    main()