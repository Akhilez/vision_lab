# Binary Bits Classifier

So in this paper called Infinity by ByteDance,
I saw that they predict d binary bits for classification.
Then they convert the d bits into a decimal number ranging from 0 to 2^d - 1.
Then they use this decimal number to classify the image.
Even if the classes (vocab size) are 10^19, the bits are still just 64.

I think this is a very interesting idea.
I want to try this out on a simple dataset like CIFAIR-10 which has 10 classes.
And then expand it to CIFAR-100 which has 100 classes and Caltech-256 which has 256 classes.
