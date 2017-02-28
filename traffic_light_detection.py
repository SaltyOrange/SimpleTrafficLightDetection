import os
import sys

import cv2
import numpy as np


class BlobDetector():

    def __init__(self, blob_color=255, min_area=8, 
        min_circularity=0.5, min_convexity=0.5, min_inertia_ratio=0.5):

        params = cv2.SimpleBlobDetector_Params()

        # Filter by Color
        params.filterByColor = True
        params.blobColor = blob_color

        # Filter by Area
        params.filterByArea = True
        params.minArea = min_area

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


class TemplateMatcher():
    
    def __init__(self, template_image_paths):
        self.templates = []
        self.initial_templates = []

        for template_image_path in template_image_paths:
            template = cv2.imread(os.path.abspath(template_image_path))
            self.initial_templates.append(template)

        self.templates = self.initial_templates.copy()

    def resize_templates(self, size):
        self.templates = []
        for initial_template in self.initial_templates:
            self.templates.append(cv2.resize(initial_template, size))

    def match_scores(self, image):
        scores = []
        for template in self.templates:
            scores.append(cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED).max())

        return scores


class KeypointCount():

    def __init__(self, keypoint, count):
        self.keypoint = keypoint
        self.count = count


class FrameProcessor():

    RED_LIGHT_TEMPLATES = ["red_light.png", "red_light2.png", "red_light3.png"]
    #GREEN_LIGHT_TEMPLATE = "light2.png"

    # BLOB_TO_TRAFFIC_LIGHT_RATIO_WIDTH = 1.5
    # BLOB_TO_TRAFFIC_LIGHT_RATIO_HEIGHT = 5.5

    RELEVANT_IMAGE_PART = (0, 180)

    BLOB_PADDING_FACTOR_UP = 0.2
    BLOB_PADDING_FACTOR_SIDE = 0.2
    BLOBS_IN_TRAFFIC_LIGHT = 3

    TEMPLATE_MATCH_THRESHOLD = 0.82

    SHOW_LIGHT_COUNT_INCREMENT = 5
    SHOW_LIGHT_COUNT_DECREMENT = 1
    SHOW_LIGHT_COUNT_LIMIT = 60
    SHOW_LIGHT_COUNT_THRESHOLD = 15

    SAME_POINT_PIXEL_TOLERANCE = 20

    LOW_FLOOD_FILL_DIFFERENCE = 30
    UP_FLOOD_FILL_DIFFERENCE = 30

 
    def __init__(self):
        self.detector = BlobDetector()
        self.red_matcher = TemplateMatcher(self.RED_LIGHT_TEMPLATES)
        #self.green_matcher = TemplateMatcher(self.GREEN_LIGHT_TEMPLATE)

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


    def check_if_same_points(self, keypoint_1, keypoint_2, pixel_tolerance):

        if keypoint_2.pt[0] - pixel_tolerance <= keypoint_1.pt[0] <= keypoint_2.pt[0] + pixel_tolerance and \
            keypoint_2.pt[1] - pixel_tolerance <= keypoint_1.pt[1] <= keypoint_2.pt[1] + pixel_tolerance:

            return True
        else:
            return False


    def draw_boxes(self, image, keypoints, color=(255, 255, 255)):

        for keypoint in keypoints:
            x, y, width, height = self.get_traffic_light_coordinates(keypoint)
            cv2.rectangle(image, (x, y), (x+width, y+height), color, 1)

        return image  


    def get_extrema_coordinates(self, image, keypoint):

        size = int(round(keypoint.size))

        x = int(round(keypoint.pt[0]))
        y = int(round(keypoint.pt[1]))

        image_part = image[y:y+size, x:x+size]

        image_part_coordinates = np.unravel_index(image_part.argmax(), image_part.shape)   

        return keypoint.pt[0]+image_part_coordinates[1]-1, keypoint.pt[1]+image_part_coordinates[0]-1


    def traffic_lights_in_frame(self, img):    

        # Store BGR image for template matching
        color_image = img.copy()

        # Convert to grayscale
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        

        # Grab only RED channel
        img = img[:, :, 2]

        # Top hat morph transform
        el = cv2.getStructuringElement(cv2.MORPH_RECT , (11, 11))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, el)
        top_hat = img - opened

        # Otsu's thresholding
        ret, top_hat = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Grab relevant frame part
        top_hat_relevant = top_hat[self.RELEVANT_IMAGE_PART[0]:self.RELEVANT_IMAGE_PART[1], :]

        # Find blobs in this frame
        # Search only in relavant frame part
        blob_keypoints = self.detector.detect(top_hat_relevant)

        im_with_keypoints = cv2.drawKeypoints(color_image, blob_keypoints, 
           np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("all keypoints", im_with_keypoints)

        # Check if filling this keypoint creates non-circular blobs
        # If it does remove this blob, else keep it and expand that keypoint to new shape
        for b_k in list(blob_keypoints):
            #cv2.imshow("pre fill %d, %d" % b_k.pt, tmp_image)
            mask = np.zeros((top_hat_relevant.shape[0]+2, top_hat_relevant.shape[1]+2), dtype=np.uint8)
            extrema_coordinates = self.get_extrema_coordinates(top_hat_relevant, b_k)

            cv2.floodFill(top_hat_relevant, mask, (int(extrema_coordinates[0]), int(extrema_coordinates[1])), 
                255, loDiff=self.LOW_FLOOD_FILL_DIFFERENCE, 
                upDiff=self.UP_FLOOD_FILL_DIFFERENCE, flags=( 255 << 8 ) | cv2.FLOODFILL_MASK_ONLY)

            blobs = self.detector.detect(mask)
            
            if 0 < len(blobs) < 2:
                blob_keypoints[blob_keypoints.index(b_k)] = blobs[0]     

                #im_with_keypoints = cv2.drawKeypoints(mask, [blobs[0]], 
                #   np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                #cv2.imshow("mask %d, %d" % b_k.pt, im_with_keypoints)
            else:
                #print("Removed %d, %d" % b_k.pt)
                #cv2.imshow("mask %d, %d" % b_k.pt, mask)
                blob_keypoints.remove(b_k)  


            #cv2.imshow("post fill %d, %d" % b_k.pt, tmp_image)              

        im_with_keypoints = cv2.drawKeypoints(img, blob_keypoints, 
            np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("all keypoints post fill filter", 
            self.draw_boxes(im_with_keypoints, blob_keypoints, (255, 0, 255)))  

        # Confirm blobs are traffic lights (template matching)
        for b_k in list(blob_keypoints):
            image_part = self.get_traffic_light_from_keypoint(color_image, b_k)

            #edges = cv2.Canny(image_part, 100, 200)
            #print(edges.shape)
            #cv2.imshow("image_part %d, %d" % b_k.pt, edges)

            self.red_matcher.resize_templates((image_part.shape[1], image_part.shape[0]))

            match_scores = self.red_matcher.match_scores(image_part)

            print("match score %d, %d" % b_k.pt, match_scores)

            if max(match_scores) < self.TEMPLATE_MATCH_THRESHOLD:
                blob_keypoints.remove(b_k) 

        im_with_keypoints = cv2.drawKeypoints(color_image, blob_keypoints, 
            np.array([]), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  

        cv2.imshow("matched blobs", self.draw_boxes(im_with_keypoints, blob_keypoints, (0, 255, 255)))  
  
        for red_light_w_count in list(self.red_lights):
            for b_k in list(blob_keypoints):               
                if self.check_if_same_points(red_light_w_count.keypoint, b_k, 
                    self.SAME_POINT_PIXEL_TOLERANCE):
                    # Check if blobs were present in previous frame in same coords
                    # and increment counter and remove that tlight from new lights
                    if red_light_w_count.count < self.SHOW_LIGHT_COUNT_LIMIT:

                        red_light_w_count.count += self.SHOW_LIGHT_COUNT_INCREMENT
                        red_light_w_count.keypoint = b_k
                        blob_keypoints.remove(b_k) 
                        break
            else:
                # Decrease count of missing traffic lights 
                red_light_w_count.count -= self.SHOW_LIGHT_COUNT_DECREMENT
                if red_light_w_count.count == 0:
                    self.red_lights.remove(red_light_w_count)


        for b_k in blob_keypoints:
            # Add new lights with count
            self.red_lights.append(KeypointCount(b_k, self.SHOW_LIGHT_COUNT_INCREMENT))

        # TODO: Determine which traffic lights to show (based on counter)
        show_lights = []
        for red_light_w_count in self.red_lights:
            if red_light_w_count.count >= self.SHOW_LIGHT_COUNT_THRESHOLD:
                show_lights.append(red_light_w_count.keypoint)

        im_with_keypoints = cv2.drawKeypoints(color_image, show_lights, 
            np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("frame", self.draw_boxes(im_with_keypoints, show_lights, (0, 0, 255)))
       

if __name__ == "__main__":

    frame_processor = FrameProcessor()

    if sys.argv[2] == "video":

        cap = cv2.VideoCapture(os.path.abspath(sys.argv[1]))    

        while cap.isOpened():
            # Capture frame-by-frame
            ret, img = cap.read()

            frame_processor.traffic_lights_in_frame(img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif sys.argv[2] == "image":
        img = cv2.imread(os.path.abspath(sys.argv[1]), cv2.IMREAD_COLOR)

        frame_processor.SHOW_LIGHT_COUNT_THRESHOLD = 0
        # Search only in upper half part of image
        frame_processor.RELEVANT_IMAGE_PART = (0, img.shape[0]//2)   

        frame_processor.traffic_lights_in_frame(img)

        cv2.waitKey()
