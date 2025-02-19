import time

import numpy as np
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import \
    postprocess_higherhrnet

last_boxes = None
last_scores = None
last_keypoints = None
drawer = None 
WINDOW_SIZE_H_W = (480, 640)
# Post-process detection threshold
THRESHOLD = 0.3

with open("../../../assets/coco_labels.txt", "r") as f:
    labels = f.read().split("\n")

def ai_output_tensor_parse(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    print(time.time())
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(outputs=np_outputs,
                                                           img_size=WINDOW_SIZE_H_W,
                                                           img_w_pad=(0, 0),
                                                           img_h_pad=(0, 0),
                                                           detection_threshold=THRESHOLD,
                                                           network_postprocess=True)

        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
    return last_boxes, last_scores, last_keypoints


def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
    """Draw the detections for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(m.array, boxes, scores,
                                  np.zeros(scores.shape), keypoints, THRESHOLD,
                                  THRESHOLD, request.get_metadata(), picam2, stream)


def parse_and_draw_pose_estimation_result(request: CompletedRequest):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    global drawer
    drawer = get_drawer()
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    ai_output_tensor_draw(request, boxes, scores, keypoints)


def get_drawer():
    categories = labels
    categories = [c for c in categories if c and c != "-"]
    return COCODrawer(categories, imx500, needs_rescale_coords=False)


if __name__ == "__main__":
    # This must be called before instantiation of Picamera2
    imx500 = IMX500("network.rpk")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(controls={'FrameRate': 30}, buffer_count=28)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = parse_and_draw_pose_estimation_result

    while True:
        time.sleep(0.5)
