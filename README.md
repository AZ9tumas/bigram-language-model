# bigram-language-model

A character-level bigram language model that learns from a list of names and generates new name-like words. Everything is in one Jupyter notebook, built two ways:

1. **Counting:** count every pair of adjacent characters (`.e`, `em`, `mm`, …) into a 27 × 27 table. Normalise each row into the probability of the next character, then sample new names from it. The model is scored with the negative log likelihood.
2. **Neural network:** the same model learned with gradient descent. Each character is one-hot encoded and passed through a single 27 × 27 linear layer and a softmax, and the weights are trained by minimising the same negative log likelihood with PyTorch autograd.

The notebook also includes a Graphviz diagram of the network's structure.

## Files

- `bigram-language-model.ipynb`: the notebook
- `names.txt`: the training data, one lowercase name per line (32,033 names)

## Requirements

- Python packages: `torch`, `matplotlib`, `graphviz`
- The Graphviz program itself, for the network diagram (`brew install graphviz` on macOS)

## Data

`names.txt` comes from Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) repository. That file is based on U.S. Social Security Administration baby-name data from [ssa.gov](https://www.ssa.gov/oact/babynames/).
