import cv2
from PIL import Image
import numpy as np
import imutils
import re
import os
import time
import random

class CaptchaHandler():

    def __init__(self):
        self.count = 0

    def convert_grey(self, file):
        # Load the image and convert it to grayscale
        self.file = file
        print(file)
        image = cv2.imread(file) # not support gif type
        print(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # array
        return gray

    def convert_binary(self, img, threshold='auto'):
        # Add some extra padding around the image
        img = cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        # threshold the image (convert it to pure black and white)
        # if threshold == 'auto': # adaptive threshold
        #     thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # cv2.THRESH_OTSU自动确定阈值
        # else:
        thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        return thresh

    def crop_img_2_letters(self, img):
        # find the contours (continuous blobs of pixels) the image
        contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Hack for compatibility with different OpenCV versions
        contours = contours[0] if imutils.is_cv2() else contours[1]

        letter_image_regions = [] # letter region
        output_array = [] # output array
        output_letters = []
        # Now we can loop through each of the four contours and extract the letter
        # inside of each one
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
            # if w / h > 1.25:
            #     # This contour is too wide to be a single letter!
            #     # Split it in half into two letter regions!
            #     half_width = int(w / 2)
            #     letter_image_regions.append((x, y, half_width, h))
            #     letter_image_regions.append((x + half_width, y, half_width, h))
            # else:
               # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

        # If we found more or less than 4 letters in the captcha, our letter extraction
        # didn't work correcly. Skip the image instead of saving bad training data!
        if len(letter_image_regions) != 4:
            print("not enough letters")
            return [],[]

        # Sort the detected letter images based on the x coordinate to make sure
        # we are processing them from left-to-right so we match the right image
        # with the right letter
        print(len(letter_image_regions))
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        captcha_correct_text = re.findall(r'(\w+)\.\w+', self.file)[0]
        # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = img[y - 2:y + h + 2, x - 2:x + w + 2] # array.shape(height, width); img.size(width, height)

            resize_img = Image.fromarray(letter_image)
            resize_img = resize_img.resize((18, 24),Image.ANTIALIAS) # change the letter image to the same size
            array = list(np.array(resize_img).flatten())
            output_array.append(array)
            output_letters.append(letter_text)
            resize_img.save('{}_{}.jpg'.format(letter_text, self.count))
            time.sleep(random.uniform(0.1, 0.3))
            self.count += 1

        return output_array, output_letters


    def multi_convert(self):
        root_path = './data/generated_captcha_images'
        files = os.listdir(root_path)
        for file in files:
            file = os.path.join(root_path, file)
            gray = self.convert_grey(file)
            binary = self.convert_binary(gray)
            _, _ = self.crop_img_2_letters(binary)

    def convert(self, file, threshold='auto'):
        gray = self.convert_grey(file)
        if threshold == 'auto':
            binary = self.convert_binary(gray)
            array, letters = self.crop_img_2_letters(binary)
        else:
            array = []
            letters = []
            threshold = 0
            while len(array) != 4:
                if threshold > 250:
                    print("max retry reached")
                    break
                threshold += 10
                binary = self.convert_binary(gray.copy(), threshold)
                array, letters = self.crop_img_2_letters(binary.copy())
        return array, letters


if __name__ == '__main__':
    ch = CaptchaHandler()
    # ch.multi_convert()
    file = 'lagou.jpg'
    ch.convert(file, threshold='manual')