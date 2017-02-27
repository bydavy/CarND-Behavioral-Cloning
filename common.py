import cv2
import matplotlib.pyplot as plt
import PIL

from PIL import Image

INPUT_IMAGE_WIDTH = int(320/2)#200)
INPUT_IMAGE_HEIGHT = int(160/2)#66)


def resize_image(image):
    """Resize image before it's feed to the neural network

        :param image: image to resize of type PIL.Image.Image
        :return: resized image
        :rtype: PIL.Image.Image
    """
    assert isinstance(image, PIL.Image.Image),\
        "Argument should be of type PIL.Image.Image"
    return image.resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),\
        Image.ANTIALIAS)
