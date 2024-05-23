# LIBMF Node

[LIBMF](https://github.com/cjlin1/libmf) - large-scale sparse matrix factorization - for Node.js

[![Build Status](https://github.com/ankane/libmf-node/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/libmf-node/actions)

## Installation

Run:

```sh
npm install libmf
```

## Getting Started

Prep your data in the format `rowIndex, columnIndex, value`

```javascript
import { Matrix } from 'libmf';

const data = new Matrix();
data.push(0, 0, 5.0);
data.push(0, 2, 3.5);
data.push(1, 1, 4.0);
```

Create a model

```javascript
import { Model } from 'libmf';

const model = new Model();
model.fit(data);
```

Make predictions

```javascript
model.predict(rowIndex, columnIndex);
```

Get the latent factors (these approximate the training matrix)

```javascript
model.p();
model.q();
```

Get the bias (average of all elements in the training matrix)

```javascript
model.bias();
```

Save the model to a file

```javascript
model.save('model.txt');
```

Load the model from a file

```javascript
const model = Model.load('model.txt');
```

Pass a validation set

```javascript
model.fit(data, evalSet);
```

Destroy the model

```javascript
model.destroy();
```

## Cross-Validation

Perform cross-validation

```javascript
model.cv(data);
```

Specify the number of folds

```javascript
model.cv(data, 5);
```

## Parameters

Pass parameters - default values below

```javascript
import { Loss } from 'libmf';

new Model({
  loss: Loss.REAL_L2,     // loss function
  factors: 8,             // number of latent factors
  threads: 12,            // number of threads used
  bins: 25,               // number of bins
  iterations: 20,         // number of iterations
  lambdaP1: 0,            // coefficient of L1-norm regularization on P
  lambdaP2: 0.1,          // coefficient of L2-norm regularization on P
  lambdaQ1: 0,            // coefficient of L1-norm regularization on Q
  lambdaQ2: 0.1,          // coefficient of L2-norm regularization on Q
  learningRate: 0.1,      // learning rate
  alpha: 1,               // importance of negative entries
  c: 0.0001,              // desired value of negative entries
  nmf: false,             // perform non-negative MF (NMF)
  quiet: false            // no outputs to stdout
});
```

### Loss Functions

For real-valued matrix factorization

- `Loss.REAL_L2` - squared error (L2-norm)
- `Loss.REAL_L1` - absolute error (L1-norm)
- `Loss.REAL_KL` - generalized KL-divergence

For binary matrix factorization

- `Loss.BINARY_LOG` - logarithmic error
- `Loss.BINARY_L2` - squared hinge loss
- `Loss.BINARY_L1` - hinge loss

For one-class matrix factorization

- `Loss.ONE_CLASS_ROW` - row-oriented pair-wise logarithmic loss
- `Loss.ONE_CLASS_COL` - column-oriented pair-wise logarithmic loss
- `Loss.ONE_CLASS_L2` - squared error (L2-norm)

## Metrics

Calculate RMSE (for real-valued MF)

```javascript
model.rmse(data);
```

Calculate MAE (for real-valued MF)

```javascript
model.mae(data);
```

Calculate generalized KL-divergence (for non-negative real-valued MF)

```javascript
model.gkl(data);
```

Calculate logarithmic loss (for binary MF)

```javascript
model.logloss(data);
```

Calculate accuracy (for binary MF)

```javascript
model.accuracy(data);
```

Calculate MPR (for one-class MF)

```javascript
model.mpr(data, transpose);
```

Calculate AUC (for one-class MF)

```javascript
model.auc(data, transpose);
```

## Resources

- [LIBMF: A Library for Parallel Matrix Factorization in Shared-memory Systems](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf_open_source.pdf)

## History

View the [changelog](https://github.com/ankane/libmf-node/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/libmf-node/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/libmf-node/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/libmf-node.git
cd libmf-node
npm install
npm run vendor
npm test
```
