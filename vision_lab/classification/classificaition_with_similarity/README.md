Traditional classification models have a fixed sized output layer.

1000 units for 1000 classes. If a new class is added, the last layer needs to be retrained at least.

What if we have a featuremap for each class that is like the centroid of all featuremaps of that class?

During training, we can use MSE loss between the featuremap of the image and the featuremap of the class.
The loss is wrt both the image featuremap and the class featuremap.

This way, we don't change the architecture of the model when a new class is added.
And possible not even need to retrain the model.

Just get the dataset for the new class,
get the average featuremap of the new class and add that as the new class featuremap.
