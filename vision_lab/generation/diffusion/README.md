Alright, I wanna create some image diffusion models.

The first paper that introduced diffusion model: https://arxiv.org/abs/2006.11239
So, the easy dataset always is of course mnist.
First let's create an unconditional diffusion model.

How diffusion works:
- Take an image x.
- Define how many steps you want to diffuse the image n.
- There's a formula that can be used to get a few things given an input image x and a noise level t.
  - A target_image that is less noisy.
  - noise z which we will predict
  - input_image which is more noisy.
- The model predicts noise z that was added at time t.
- Given a more noisy image and the noise z, we can get a less noisy image using the formula.

For this simple experiment, I'll use a simple feed forward model 
that takes 28*28=784 (pixels) + 1 (time step) as input and predicts 784 (pixels) as output.

Once this works, we can update to generating specific classes.