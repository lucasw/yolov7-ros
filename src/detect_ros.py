#!/usr/bin/python3

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.ros import create_detection_msg
from visualizer import draw_detections

import os
from typing import Tuple, Union

import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rospy

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape
    copied from https://github.com/meituan/YOLOv6/blob/main/yolov6/core/inferer.py
    '''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (
                ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes


class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = attempt_load(weights, map_location=device)

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections


class Yolov7Publisher:
    def __init__(self, weights: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45,
                 device: str = "cuda",
                 queue_size: int = 1, visualize: bool = False):
        """
        :param weights: path/to/yolo_weights.pt
        :param conf_thresh: confidence threshold
        :param iou_thresh: intersection over union threshold
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        """
        self.device = device

        self.visualization_publisher = rospy.Publisher(
            "viz_image", Image, queue_size=queue_size
        ) if visualize else None

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        self.model = YoloV7(
            weights=weights, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
            device=device
        )
        self.img_subscriber = rospy.Subscriber(
            "image", Image, self.img_callback
        )
        self.detection_publisher = rospy.Publisher(
            "detection2d", Detection2DArray, queue_size=queue_size
        )

    def img_callback(self, img_msg: Image):
        t0 = rospy.Time.now()
        self.process_img_msg(img_msg)
        elapsed = rospy.Time.now() - t0
        rospy.loginfo(f"{elapsed.to_sec():0.3f}")

    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """

        # automatically resize the image to the next smaller possible size
        # w_scaled, h_scaled = self.img_size

        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough'
        )
        h_orig, w_orig, c = np_img_orig.shape

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        if c == 1:
            np_img_orig = np.concatenate([np_img_orig] * 3, axis=2)
            c = 3

        # if True:
        #     # w_scaled = w_orig - (w_orig % 8)
        #     # This line seems to result in a memory leak
        #     # np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))
        #     # np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled), interpolation=cv2.INTER_LINEAR)
        #     np_img_resized = np_img_orig[:w_scaled, :h_scaled, :]
        #     rospy.loginfo(np_img_resized.shape)
        # else:
        #     np_img_resized = np.zeros((w_scaled, h_scaled, 3), dtype=np.uint8)
        np_img_resized = np_img_orig

        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)

        # inference & rescaling the output to original img size
        detections = self.model.inference(img)

        # detections[:, :4] = rescale([h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()

        # publishing
        detection_msg = create_detection_msg(img_msg, detections)
        self.detection_publisher.publish(detection_msg)

        # visualizing if required
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            vis_img = draw_detections(np_img_orig, bboxes, classes)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
            vis_msg.encoding = img_msg.encoding
            self.visualization_publisher.publish(vis_msg)


if __name__ == "__main__":
    rospy.init_node("yolov7_node")

    weights_path = rospy.get_param("~weights_path")
    conf_thresh = rospy.get_param("~conf_thresh")
    iou_thresh = rospy.get_param("~iou_thresh")
    queue_size = rospy.get_param("~queue_size")
    visualize = rospy.get_param("~visualize")
    device = rospy.get_param("~device")

    # some sanity checks
    if not os.path.isfile(weights_path):
        raise FileExistsError("Weights not found.")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")

    publisher = Yolov7Publisher(
        weights=weights_path,
        device=device,
        visualize=visualize,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        queue_size=queue_size
    )

    rospy.spin()
