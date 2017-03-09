import os
import sys
import argparse

import cv2
import numpy as np
from sklearn.externals import joblib


class BlobDetector():

    def __init__(self, blob_color=255, min_area=16, max_area=38400,
        min_circularity=0.6, min_convexity=0.6, min_inertia_ratio=0.5):

        params = cv2.SimpleBlobDetector_Params()

        # Filter by Color
        params.filterByColor = True
        params.blobColor = blob_color

        # Filter by Area
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = min_circularity

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = min_convexity

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia_ratio

        self.detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, image):
        return self.detector.detect(image)


class TrafficLightClassifier():    

    def __init__(self, pca_path, clf_path, image_dims=(20, 60), image_channels=3):
        self.image_dims = image_dims
        self.image_channels = image_channels
        self.flattened_dims = self.image_dims[0] * self.image_dims[1] * self.image_channels

        self.pca = joblib.load(os.path.abspath(pca_path))
        self.clf = joblib.load(os.path.abspath(clf_path))

    def predict(self, img):

        return self.clf.predict(
            self.pca.transform([cv2.resize(img, self.image_dims).reshape(self.flattened_dims)])
        )


class TrafficLightWCount():

    def __init__(self, keypoint, count=None, increment=5, 
            decrement=1, threshold=15, lower_bound=0, upper_bound=60):

        self.keypoint = keypoint
        self.count = count if count else increment
        self.increment = increment
        self.decrement = decrement
        self.threshold = threshold
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def increment_count(self):
        if self.count + self.increment > self.upper_bound:
            self.count = self.upper_bound
        else:
            self.count += self.increment

    def decrement_count(self):
        if self.count - self.decrement < self.lower_bound:
            self.count = self.lower_bound
        else:
            self.count -= self.decrement

    def gteq_threshold(self):
        return self.count >= self.threshold


class FrameProcessor():

    RELEVANT_IMAGE_PART = (0, 180)

    BLOB_PADDING_FACTOR_UP = 0.2
    BLOB_PADDING_FACTOR_SIDE = 0.2
    BLOBS_IN_TRAFFIC_LIGHT = 3

    SAME_POINT_PIXEL_TOLERANCE = 20

    RED_LIGHT_CLASS = 0

 
    def __init__(self):
        self.detector = BlobDetector()
        # TODO: Make green light logic and classifier
        self.classifier = TrafficLightClassifier("models/pca.pkl", "models/svm.pkl")

        self.red_lights = []        

    def get_traffic_light_coordinates(self, keypoint):

        # -----
        # | O | <-- Add padding to traffic light (left and right of the light itself, 2 * PADDING)
        # | O | <-- And padding above every light (3 * PADDDING)
        # | O | <-- Also add 1 PADDING below last light
        # -----

        # TODO: Horizontal lights processing and green light processing

        width = int(round(keypoint.size + 2 * self.BLOB_PADDING_FACTOR_SIDE * keypoint.size))
        height = int(round((keypoint.size + 2 * self.BLOB_PADDING_FACTOR_UP * keypoint.size) \
            * self.BLOBS_IN_TRAFFIC_LIGHT))

        x = max(0, int(round(keypoint.pt[0] - (self.BLOB_PADDING_FACTOR_SIDE + 0.5) * keypoint.size)))
        y = max(0, int(round(keypoint.pt[1] - (self.BLOB_PADDING_FACTOR_UP + 0.5) * keypoint.size)))        

        return x, y, width, height

    def get_traffic_light_from_keypoint(self, image, keypoint):        

        x, y, width, height = self.get_traffic_light_coordinates(keypoint)

        return image[y:y+height, x:x+width]

    def draw_boxes(self, image, keypoints, color=(255, 255, 255)):

        for keypoint in keypoints:
            x, y, width, height = self.get_traffic_light_coordinates(keypoint)
            cv2.rectangle(image, (x, y), (x+width, y+height), color, 1)

        return image  

    def check_if_same_points(self, keypoint_1, keypoint_2, pixel_tolerance):

        if keypoint_2.pt[0] - pixel_tolerance <= keypoint_1.pt[0] <= keypoint_2.pt[0] + pixel_tolerance and \
            keypoint_2.pt[1] - pixel_tolerance <= keypoint_1.pt[1] <= keypoint_2.pt[1] + pixel_tolerance:

            return True
        else:
            return False   

    def traffic_lights_in_frame(self, img):    

        # Store RGB image for template matching
        color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # RED "normalised" channel
        img = img.astype(np.int16)
        img = np.clip(img[:, :, 2] - (img[:, :, 0] + img[:, :, 1]/2)/2, 0, 255)
        img = img.astype(np.uint8)

        # Grab relevant frame part
        img_relevant = img[self.RELEVANT_IMAGE_PART[0]:self.RELEVANT_IMAGE_PART[1], ...]

        # Find blobs in this frame
        # Search only in relavant frame part
        blob_keypoints = self.detector.detect(img_relevant)

        # Confirm blobs are traffic lights
        blob_keypoints = filter(
            lambda blob: self.classifier.predict(
                self.get_traffic_light_from_keypoint(color_image, blob)) == self.RED_LIGHT_CLASS,
            blob_keypoints
        )

        for red_light_w_count in list(self.red_lights):
            is_incremented = False       
            for b_k in list(blob_keypoints):        
                if self.check_if_same_points(red_light_w_count.keypoint, b_k, 
                        self.SAME_POINT_PIXEL_TOLERANCE):

                    # Check if blobs were present in previous frame in same 
                    # coords and increment counter and remove that light 
                    # from new lights
                    if not is_incremented: 
                        red_light_w_count.increment_count()
                        is_incremented = True

                    red_light_w_count.keypoint = b_k
                    blob_keypoints.remove(b_k) 

            if not is_incremented:
                # Decrease count of missing traffic lights 
                red_light_w_count.decrement_count()
                if red_light_w_count.count == 0:
                    self.red_lights.remove(red_light_w_count)

        # Add new lights with sufficient count
        self.red_lights.extend(
            [TrafficLightWCount(b_k) for b_k in blob_keypoints]
        )

        # Determine which traffic lights to show (based on counter)
        show_lights = [rl.keypoint for rl in filter(
            lambda rlc: rlc.gteq_threshold, 
            self.red_lights
        )]

        return cv2.cvtColor(self.draw_boxes(
            cv2.drawKeypoints(color_image, show_lights, np.array([]), 
                (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), 
            show_lights,
            color=(255, 0, 255)
        ), cv2.COLOR_RGB2BGR)
       

parser = argparse.ArgumentParser(
    description="Traffic light detection and localization (Currently only HORIZONTAL RED lights)\n"
)

file_type_choices = ["image", "video"]

parser.add_argument("file_type", type=str, choices=file_type_choices,
                    help="Specify what kind of file you want to process")
parser.add_argument("file_path", type=str, help="Path to file to be processed")

args = parser.parse_args()

if __name__ == "__main__":

    frame_processor = FrameProcessor()

    if args.file_type == file_type_choices[1]:

        cap = cv2.VideoCapture(os.path.abspath(args.file_path))    

        while cap.isOpened():
            # Capture frame-by-frame
            ret, img = cap.read()

            if ret:
                result = frame_processor.traffic_lights_in_frame(img)
                cv2.imshow("Result", result)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.file_type == file_type_choices[0]:
        img = cv2.imread(os.path.abspath(args.file_path), cv2.IMREAD_COLOR)

        frame_processor.SHOW_LIGHT_COUNT_THRESHOLD = 0
        # Specify part of image to search in
        frame_processor.RELEVANT_IMAGE_PART = (0, img.shape[0]) 

        result = frame_processor.traffic_lights_in_frame(img)
        cv2.imshow("Result", result)

        cv2.waitKey()
