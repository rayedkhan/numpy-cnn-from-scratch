# A CNN in NumPy

A convolutional network that reads MNIST digits, with every operation written by hand: the convolution, the pooling, the softmax, the gradients, the weight updates. No autograd, no framework. The point was to see what a CNN actually is once the abstractions are stripped away.

```
28x28 image
  -> conv, 2 filters of 3x3, sigmoid  ->  2 x 26x26
  -> average pool 2x2, stride 2       ->  2 x 13x13
  -> flatten                          ->  338
  -> dense                            ->  10 logits -> softmax
```

## Result

Ten epochs over the 60,000 training images at a learning rate of 0.01. Training loss fell from 0.0503 to 0.0337 and flattened out after the fourth epoch. Accuracy on the 10,000 held-out test images: **88.19%**. The seed is fixed in `initialize_weights`, so a rerun reproduces those numbers exactly.

## What it showed

Writing backpropagation by hand shows you exactly what you did and did not derive. The dense weights update on the true gradient of the loss. The convolutional kernels do not: their update is the activation derivative convolved with the input, which never chains back through the pooling layer and never sees the label at all. The network reaches 88.19% anyway, which is the uncomfortable part. Shapes line up and loss goes down whether or not the maths behind them is right.

## Run it

NumPy is the only dependency.

```bash
pip install numpy jupyter
jupyter notebook numpy-cnn.ipynb
```

Run the cells top to bottom. The first run pulls `mnist.npz` (11 MB) next to the notebook and caches it there. If that download fails with an SSL certificate error, which some Python installs on macOS do, fetch it yourself first:

```bash
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Training is a Python loop over one image at a time, so all ten epochs take roughly an hour on a laptop. Slice `x_train` down to a few thousand images to watch it work in under a minute.
