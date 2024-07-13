import ffi, { MfModel } from './ffi.js';
import koffi from 'koffi';
import path from 'path';

export default class Model {
  constructor(options = {}) {
    this.options = options;
  }

  fit(data, evalSet) {
    const trainSet = this.#createProblem(data);

    let model;
    if (evalSet) {
      evalSet = this.#createProblem(evalSet);
      model = ffi.mf_train_with_validation(trainSet, evalSet, this.#param());
    } else {
      model = ffi.mf_train(trainSet, this.#param());
    }

    if (model === null) {
      throw new Error('fit failed');
    }

    this.#setModel(model);
  }

  predict(rowIndex, columnIndex) {
    return ffi.mf_predict(this.#modelPtr(), rowIndex, columnIndex);
  }

  cv(data, folds = 5) {
    const problem = this.#createProblem(data);
    // TODO update fork to differentiate between bad parameters and zero error
    const res = ffi.mf_cross_validation(problem, folds, this.#param());
    if (res === 0) {
      throw new Error('cv failed');
    }
    return res;
  }

  save(path) {
    const status = ffi.mf_save_model(this.#modelPtr(), path);
    if (status !== 0) {
      throw new Error('Cannot save model');
    }
  }

  static load(path) {
    const model = new Model();
    model.loadModel(path);
    return model;
  }

  loadModel(path) {
    const model = ffi.mf_load_model(path);
    if (model === null) {
      throw new Error('Cannot open model');
    }
    this.#setModel(model);
  }

  rows() {
    return this.#model().m;
  }

  columns() {
    return this.#model().n;
  }

  factors() {
    return this.#model().k;
  }

  bias() {
    return this.#model().b;
  }

  p() {
    return this.#readFactors(this.#model().p, this.rows());
  }

  q() {
    return this.#readFactors(this.#model().q, this.columns());
  }

  rmse(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_rmse(prob, this.#modelPtr());
  }

  mae(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_mae(prob, this.#modelPtr());
  }

  gkl(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_gkl(prob, this.#modelPtr());
  }

  logloss(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_logloss(prob, this.#modelPtr());
  }

  accuracy(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_accuracy(prob, this.#modelPtr());
  }

  mpr(data, transpose) {
    const prob = this.#createProblem(data);
    return ffi.calc_mpr(prob, this.#modelPtr(), transpose);
  }

  auc(data, transpose) {
    const prob = this.#createProblem(data);
    return ffi.calc_auc(prob, this.#modelPtr(), transpose);
  }

  // TODO do this automatically
  destroy() {
    this.#destroyModel();
  }

  #readFactors(ptr, n) {
    const f = this.factors();
    const rowSize = koffi.sizeof(koffi.types.float) * f;
    const factors = [];
    for (let i = 0; i < n; i++) {
      factors.push(koffi.decode(ptr, i * rowSize, koffi.types.float, f));
    }
    return factors;
  }

  #checkFit() {
    if (!this.modelPtr) {
      throw new Error('Not fit');
    }
  }

  #model() {
    this.#checkFit();
    return this.model;
  }

  #modelPtr() {
    this.#checkFit();
    return this.modelPtr;
  }

  #setModel(modelPtr) {
    this.#destroyModel();
    this.model = koffi.decode(modelPtr, MfModel);
    this.modelPtr = modelPtr;
  }

  #destroyModel() {
    if (this.modelPtr) {
      ffi.mf_destroy_model([this.modelPtr]);
      this.modelPtr = null;
      this.model = null;
    }
  }

  #param() {
    const param = ffi.mf_get_default_param();
    param.copy_data = false;

    // silence insufficient blocks warning with default params
    param.nr_bins = 25;

    const options = this.options;
    if ('loss' in options) {
      param.fun = options.loss;
    }
    if ('factors' in options) {
      param.k = options.factors;
    }
    if ('threads' in options) {
      param.nr_threads = options.threads;
    }
    if ('bins' in options) {
      param.nr_bins = options.bins;
    }
    if ('iterations' in options) {
      param.nr_iters = options.iterations;
    }
    if ('lambdaP1' in options) {
      param.lambda_p1 = options.lambdaP1;
    }
    if ('lambdaP2' in options) {
      param.lambda_p2 = options.lambdaP2;
    }
    if ('lambdaQ1' in options) {
      param.lambda_q1 = options.lambdaQ1;
    }
    if ('lambdaQ2' in options) {
      param.lambda_q2 = options.lambdaQ2;
    }
    if ('learningRate' in options) {
      param.eta = options.learningRate;
    }
    if ('alpha' in options) {
      param.alpha = options.alpha;
    }
    if ('c' in options) {
      param.c = options.c;
    }
    if ('nmf' in options) {
      param.do_nmf = options.nmf;
    }
    if ('quiet' in options) {
      param.quiet = options.quiet;
    }

    // do_nmf must be true for generalized KL-divergence
    if (param.fun === 2) {
      param.do_nmf = true;
    }

    return param;
  }

  #createProblem(data) {
    if (typeof data === 'string') {
      // need to expand path so it's absolute
      return ffi.mf_read_problem(path.resolve(data));
    }

    data = data.data;

    if (data.length === 0) {
      throw new Error('No data');
    }

    let m = 0;
    let n = 0;

    for (let row of data) {
      if (row.u >= m) {
        m = row.u + 1;
      }

      if (row.v >= n) {
        n = row.v + 1;
      }
    }

    const prob = {};
    prob.m = m;
    prob.n = n;
    prob.nnz = data.length;
    prob.r = data;
    return prob;
  }
};
