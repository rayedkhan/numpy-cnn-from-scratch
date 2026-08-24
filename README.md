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

Ten epochs over the 60,000 training images at a learning rate of 0.01, one image at a time. Training loss fell from 0.507 to 0.128 and was still dropping when it stopped. Accuracy on the 10,000 held-out test images: **96.29%**. The seed is fixed in `initialize_weights`, so a rerun reproduces those numbers exactly.

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

Training is a Python loop over one image at a time, so all ten epochs take about 35 minutes on a laptop. Slice `x_train` down to a few thousand images to watch it work in under a minute.
