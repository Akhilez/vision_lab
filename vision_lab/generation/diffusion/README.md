Alright, I wanna create some image diffusion models.

So, the easy dataset always is of course mnist.
The input would be the image label as text and the output would be a 28x28 image.

Next, maybe I can work on inpainting an odometer bbox.
Cool, so let's get the mnist working.

---

For training, you might need these tuples:
(
    image with noisy region,
    level of noise,
    text,
    noise label,
    image identifier,
)
