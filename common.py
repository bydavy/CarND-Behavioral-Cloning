import PIL

from PIL import Image

# top area that will be cropped (above horizon)
top_crop = 52
# bottom area that will be cropped (car's hood)
bottom_crop = 25

input_image_width = int(320 / 3)
input_image_height = int((160 - top_crop - bottom_crop) / 3)


def resize_image(image):
    """Resize image before it's feed to the neural network

        :param image: image to resize of type PIL.Image.Image
        :return: resized image
        :rtype: PIL.Image.Image
    """
    assert isinstance(image, PIL.Image.Image), "Argument should be of type PIL.Image.Image"
    # As I'm training on CPU, I took drastic cropping to reduce the number of pixels to process
    # In real life, we cannot be so drastic: horizon is not at a fixed position in the image (steep road, etc)
    image = image.crop(box=(0, top_crop, 320, 160 - bottom_crop))
    return image.resize((input_image_width, input_image_height), Image.ANTIALIAS)
