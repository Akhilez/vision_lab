# Style Transfer

This is such a fun idea.
The idea is to take a latent noise and backprop using error b/w 
1. feature maps of a target image and 
2. a correlation matrix of a style image.

Get a sample style and target image.

First, let's just do the first part -- backpropping using feature map of a target image.
So we need a pre-trained CNN that we can use to extract feature maps at different layers.
The latent and the target image are the inputs to the CNN.
The outputs are the feature maps at different layers.
Then we write our own loss function to minimize the MSE b/w the feature maps wrt the latent image.
We can achieve this by setting requires_grad=True for the latent image.
And the optimizer is defined to use the latent image as the parameter to optimize, not the CNN's parameters.
After a step of optimization, the gradients are added to the latent image by the optimizer.
Save this image and visualize it later as an animation.
Do this some n times and then look at the animation.

Next, we can do the same, but instead of applying MSE on the raw feature maps,
we create a correlation matrix of the feature maps and apply MSE on that.
The correlation matrix is simply a matrix multiplication of the feature maps with their transpose.
Let's again see how the animation looks like.

Finally, we can combine the two losses and see how the animation looks like.
