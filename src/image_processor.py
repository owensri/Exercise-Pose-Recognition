import cv2

class ImageProcessor:
    def convert_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def flip_horizontal(self, image):
        return cv2.flip(image, 1)