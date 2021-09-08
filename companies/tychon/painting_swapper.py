from os import listdir
from os.path import join
from random import randint
import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


def swap_paintings(images_path):
    for image_name in tqdm(listdir(images_path)):
        image = Image.open(join(images_path, image_name))
        image = np.array(image, np.uint8)

        # Finding the paintings
        edges = cv2.Canny(image, 50, 100)
        thresholded = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 35)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contours = [contour for contour in contours if is_painting(contour)]
        best_contours = [cv2.convexHull(contour) for contour in best_contours]

        # Replacing the paintings with puppies
        puppies = np.zeros_like(image)
        for contour in best_contours:
            x1, y1, w, h = cv2.boundingRect(contour)
            puppies[y1: y1 + h, x1: x1 + w] = get_random_image(h, w, 'puppy')

        mask = np.zeros_like(image, np.uint8)
        cv2.drawContours(mask, best_contours, -1, (255, 255, 255), -1)

        puppies = puppies & mask
        image = image & ~mask
        image += puppies

        Image.fromarray(image).save(f'output/{image_name}')


def is_painting(contour):
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if aspect_ratio < 0.3 or aspect_ratio > 7:
        return False

    area = cv2.contourArea(contour)
    if area < 8000:
        return False

    rect_area = w * h
    rectangle_ratio = area / rect_area
    if rectangle_ratio < 0.6:
        return False

    return True


def get_random_image(h, w, keyword):
    try:
        url = f'https://loremflickr.com/480/320/{keyword}'
        img = Image.open(requests.get(url, stream=True).raw)
        img = img.convert('RGB')
        img = np.array(img, dtype=np.uint8)
        img = cv2.resize(img, (w, h))
        return img
    except Exception as e:
        print(e)
        return np.ones((h, w, 3)) * randint(0, 255)


if __name__ == '__main__':
    images_path = './data'
    swap_paintings(images_path)
