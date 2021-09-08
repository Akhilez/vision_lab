from os import listdir, makedirs
from os.path import join
from random import randint
import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


def swap_paintings(images_path: str, output_path: str):
    """
    Swaps paintings and saves them to output dir
    :param images_path: directory path to input images
    :param output_path: directory path to output images
    :return: None
    """
    makedirs(output_path, exist_ok=True)
    for image_name in tqdm(listdir(images_path)):
        image = Image.open(join(images_path, image_name))
        image = np.array(image, np.uint8)

        contours = find_paintings(image)
        output_image = replace_with_puppies(image, contours)

        Image.fromarray(output_image).save(join(output_path, image_name))


def replace_with_puppies(image: np.ndarray, contours: list) -> np.ndarray:
    """
    Replaces the contours with puppy images.

    Assumption: Resized puppy images = puppy image with modified perspective.

    1. In a blank image, place all the puppies to fill the painting bounding boxes.
    2. Create a mask image by filling the contours.
    3. Apply mask intersection with puppies
    4. Apply intersection b/w image and inverse mask so that we get all the background.
    5. Now add background and puppies

    :param image: input image of shape (h, w, 3)
    :param contours: opencv contours.
    :return: swapped image of shape (h, w, 3)
    """
    # Replacing the paintings with puppies
    puppies = np.zeros_like(image)
    for contour in contours:
        x1, y1, w, h = cv2.boundingRect(contour)
        puppies[y1: y1 + h, x1: x1 + w] = get_random_image(h, w, 'puppy')

    mask = np.zeros_like(image, np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    puppies = puppies & mask
    image = image & ~mask
    image += puppies

    return image


def find_paintings(image: np.ndarray) -> list:
    """
    Finding the paintings

    1. Canny edge detection
    2. [Adaptive Gaussian Thresholding](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
    3. Find contours
    4. Filter the contours
    5. Apply convel hull

    All hyper-parameters are empirically found.

    :param image: input image of shape (h, w, 3)
    :return: opencv contours
    """
    edges = cv2.Canny(image, 50, 100)
    thresholded = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 35)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contours = [contour for contour in contours if is_painting(contour)]
    best_contours = [cv2.convexHull(contour) for contour in best_contours]
    return best_contours


def is_painting(contour: list) -> bool:
    """
    Predicate used to filter the painting contours.

    Assumptions:
    - aspect ratio of bounding box has to be in a certain range.
    - area of the contour has to be greater than a threshold.
    - contour's area / bounding box area has to be within a range.

    The contours are filtered based on the assumptions above.
    The actual ranges and thresholds are empirically found.

    :param contour: an opencv contour.
    :return: True if painting else False.
    """
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


def get_random_image(h: int, w: int, keyword: str) -> np.array:
    """
    Returns a random image from flickr with search term provided.
    Returns a blank random colored image if fetch fails.

    :param h: height of desired image
    :param w: width of desired image
    :param keyword: search term
    :return: image as np.ndarray of shape (h, w, 3)
    """
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
    output_path = './output'

    swap_paintings(images_path, output_path)
