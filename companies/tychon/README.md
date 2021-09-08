# Tychon assignment submission
_**Akhil Devarashetti**_

The final code to generate puppy images is in `painting_swapper.py`.
The outputs are in `output` directory.
Please note that the puppy images are extracted randomly from flickr,
so some of them are not the best ones.

I used classical CV approach which can be seen in `experiment_1.ipynb`.

I also tried a deep learning approach with synthetic data - seen in `experiment_2.ipynb`

### Takeaways

This was a fun problem.
I think the classical approach can be further optimized with creative methods and hyper params tuning.

The deep learning approach can be significantly improved by

- real training data
- direct (x, y) prediction for each corner of each painting (quadrilateral detection).

---

*This is a ML/AI - Computer Vision take-home style assignment for tychon candidates*

# Painting Swap

## 1. The Problem

The problem is to locate paintings and swap them for cute puppy pictures. 

Develop a Computer Vision program which takes in a set images. For each image you are required to locate each of the paintings within the images, and swap each painting with a random puppy image in its place. The swapped image must sit within the painting frame as if it were the original painting.  

![Example 1](https://bitbucket.org/kevinbt/swap-paintings/raw/c2243164d88a067513ac707040554ecc5f97efc9/assets/swap-example-1.jpg)

The walls of the galleries in the images are assumed to be a single colour, as is usual in an Art gallery so as to not distract from the art. This assumption helps to detect the non-wall components of the gallery image.  

![Example 2](https://bitbucket.org/kevinbt/swap-paintings/raw/c2243164d88a067513ac707040554ecc5f97efc9/assets/swap-example-2.jpg)

The paintings are also assumed to be somewhat rectangular (*somewhat* specifically meaning that they take up 60% of their bounding rectangle in the image). This assumption helps to eliminate parts of the wall and ceiling that may be in the gallery image.

## 2. Submission Instructions

Using the images in the data folder, complete the challenage by generating your replaced images in the output folder. 

Commit your final version to your own Github repository. 

Email your repository link containing your solution.
