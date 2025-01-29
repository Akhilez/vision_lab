Paper link: https://arxiv.org/pdf/1711.00937
Try it on CIFAR10 dataset.

This is further explored in Infinity VAR paper by ByteDance.
There is a bit correction part in the paper.
They do
1. They randomly flip bits that are sent to decoder with a probability p=0.3
2. They use a bit correction network to correct the predicted bits. This uses the context to correct the bits.
   1. pred -> quantize -> correction network -> pred -> quantize

Just realized this is totally different from what the paper did.
The paper did not add any extra computations to correct the bits.
The bit correction is part of the residual creation process.
We encode F (the raw feature map) into a bunch of Rs.
The previous R will have some bits that are wrong.
But next R is somehow created in a way that accounts of the wrong bits in previous R.
Not well understood at this point, but yeah, totally different from what I was thinking.
