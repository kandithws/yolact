//
// Created by kandithws on 2/5/2562.
//

#include <iostream>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include "detection_v2.grpc.pb.h"


class PredictedObject {
  public:
    enum class MASK_TYPE : int {
        NO_MASK=-1,
        CROPPED=0,
        FULL=1
    };
    PredictedObject(const int& label, const float& conf, const cv::Rect& box);
    PredictedObject(const int& label, const float& conf, const cv::Rect& box,
            const cv::Mat &mask, const MASK_TYPE& mask_type=MASK_TYPE::CROPPED);

    const int _label;
    const float _confidence;
    const cv::Mat _mask;
    const MASK_TYPE _mask_type;
    const cv::Rect2f _bbox;

    inline cv::Point2f GetBoxCentroid() const {
        return (_bbox.tl() + _bbox.br()) * 0.5;
    }
};

PredictedObject::PredictedObject(const int &label, const float &conf, const cv::Rect &box)
: _label(label), _confidence(conf), _bbox(box), _mask_type(MASK_TYPE::NO_MASK)
{}


PredictedObject::PredictedObject(const int &label, const float& conf, const cv::Rect& box,
                const cv::Mat &mask, const MASK_TYPE& mask_type) :
                _label(label), _confidence(conf), _bbox(box),
                _mask_type(mask_type),
                _mask(mask.clone())
                {}


class DetectionV2Client {
  public:
    DetectionV2Client(std::shared_ptr<grpc::Channel> channel)
            : stub_(detection_service_v2::InstanceDetectionService::NewStub(channel)) {}

    detection_service_v2::Image toImageRequest(const cv::Mat &img, bool rgb = true) {
        auto img_msg = detection_service_v2::Image();
        // TODO -- assert CV Type,
        img_msg.set_channel(3);
        size_t size = img.total() * img.elemSize();
        img_msg.set_data(img.data, size * sizeof(char));
        img_msg.set_height(img.rows);
        img_msg.set_width(img.cols);
        img_msg.set_type("uint8"); // TO FIX
        return img_msg;
    };


    bool detectObject(const cv::Mat &img, detection_service_v2::InstanceDetections &out) {
        detection_service_v2::InstanceDetections response;
        auto req_img = toImageRequest(img);
        grpc::ClientContext context;
        auto st = stub_->DetectInstances(&context, req_img, &response);

        if (st.ok()) {
            response = out;
            return true;
        } else {
            std::cout << "ERROR! " << std::endl;
            std::cout << st.error_message() << std::endl;
            std::cout << st.error_details() << std::endl;
            return false;
        }
    }

  private:
    std::unique_ptr<detection_service_v2::InstanceDetectionService::Stub> stub_;
};

int main(int argc, char **argv) {
    DetectionV2Client image_client(grpc::CreateChannel(
            "0.0.0.0:50051", grpc::InsecureChannelCredentials()));

    if (argc != 2) {
        std::cout << "Usage: image_client [image_file]" << std::endl;
        return -1;
    }
    cv::Mat img = cv::imread(argv[1]);
    detection_service_v2::InstanceDetections outs;

    if (image_client.detectObject(img, outs)) {
        std::vector<std::shared_ptr<PredictedObject> > preds;
        preds.reserve(outs.predictions_size());
        for (auto &out : outs.predictions()){
            int label_id = out.label_id();
            float conf = out.confidence();
            cv::Rect2f box(cv::Point2f(out.box().tlx(), out.box().tly()),
                    cv::Point2f(out.box().brx(), out.box().bry()));

            PredictedObject::MASK_TYPE mt(static_cast<PredictedObject::MASK_TYPE>(out.mask_type()));
            auto buffer = out.mask().data().c_str();
            cv::Mat mask(out.mask().height(), out.mask().width(), CV_8UC1, &buffer);
            preds.push_back(std::make_shared<PredictedObject>(
                    label_id, conf, box, mask, mt
                    ));
        }
    } else {
        std::cout << "Not connected" << std::endl;
    }
    return 0;
}
