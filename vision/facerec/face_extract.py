import cv
import os


class FaceExtractor(object):

    CASCADE_FILE_PATH = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'

    def extract_faces(self, image_path, haar_scale=1.2, min_neighbors=5, min_width=50):
        assert os.path.exists(image_path)
        min_size = (min_width, min_width)
        image = cv.LoadImage(image_path,  cv.CV_LOAD_IMAGE_GRAYSCALE)
        haar = cv.Load(self.CASCADE_FILE_PATH)
        detected = cv.HaarDetectObjects(image, haar, cv.CreateMemStorage(), haar_scale, min_neighbors, cv.CV_HAAR_DO_CANNY_PRUNING, min_size)
        # cvHaarDetecObjects returns list of tuples [ ((x, y, width, height), min_neighbors), ... ]
        # but we ignore the height and return a square box of length 'width'
        return [(x, y, w, w) for (x, y, w, h), n in detected]
