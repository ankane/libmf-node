import ffi, { Node, Problem } from './ffi.js';
import path from 'path';
import ref from 'ref-napi';

export default class Model {
  constructor(options = {}) {
    this.options = options;
  }

  fit(data, evalSet) {
    const trainSet = this.#createProblem(data);

    let model;
    if (evalSet) {
      evalSet = this.#createProblem(evalSet);
      model = ffi.mf_train_with_validation(trainSet.ref(), evalSet.ref(), this.#param());
    } else {
      model = ffi.mf_train(trainSet.ref(), this.#param());
    }

    if (model.isNull()) {
      throw new Error('fit failed');
    }

    this.#setModel(model);
  }

  predict(rowIndex, columnIndex) {
    return ffi.mf_predict(this.#model(), rowIndex, columnIndex);
  }

  cv(data, folds = 5) {
    const problem = this.#createProblem(data);
    // TODO update fork to differentiate between bad parameters and zero error
    const res = ffi.mf_cross_validation(problem.ref(), folds, this.#param());
    if (res === 0) {
      throw new Error('cv failed');
    }
    return res;
  }

  save(path) {
    const status = ffi.mf_save_model(this.#model(), path);
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
    if (model.isNull()) {
      throw new Error('Cannot open model');
    }
    this.#setModel(model);
  }

  rows() {
    return this.#model().deref().m;
  }

  columns() {
    return this.#model().deref().n;
  }

  factors() {
    return this.#model().deref().k;
  }

  bias() {
    return this.#model().deref().b;
  }

  p() {
    return this.#readFactors(this.#model().deref().p, this.rows());
  }

  q() {
    return this.#readFactors(this.#model().deref().q, this.columns());
  }

  rmse(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_rmse(prob.ref(), this.#model());
  }

  mae(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_mae(prob.ref(), this.#model());
  }

  gkl(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_gkl(prob.ref(), this.#model());
  }

  logloss(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_logloss(prob.ref(), this.#model());
  }

  accuracy(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_accuracy(prob.ref(), this.#model());
  }

  mpr(data, transpose) {
    const prob = this.#createProblem(data);
    return ffi.calc_mpr(prob.ref(), this.#model(), transpose);
  }

  auc(data, transpose) {
    const prob = this.#createProblem(data);
    return ffi.calc_auc(prob.ref(), this.#model(), transpose);
  }

  // TODO do this automatically
  destroy() {
    this.#destroyModel();
  }

  // TODO improve performance
  #readFactors(ptr, n) {
    const f = this.factors();
    const buf = ptr.ref().readPointer(0, n * f * 4);
    const factors = [];
    let offset = 0;
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < f; j++) {
        row.push(ref.types.float.get(buf, offset));
        offset += 4;
      }
      factors.push(new Float32Array(row));
    }
    return factors;
  }

  #model() {
    if (!this.model) {
      throw new Error('Not fit');
    }
    return this.model;
  }

  #setModel(model) {
    this.#destroyModel();
    this.model = model;
  }

  #destroyModel() {
    if (this.model) {
      ffi.mf_destroy_model(this.model.ref());
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
    const r = Buffer.allocUnsafe(Node.size * data.length);
    let offset = 0;

    for (let row of data) {
      ref.types.int.set(r, offset, row[0]);
      offset += 4;
      ref.types.int.set(r, offset, row[1]);
      offset += 4;
      ref.types.float.set(r, offset, row[2]);
      offset += 4;

      if (row[0] >= m) {
        m = row[0] + 1;
      }

      if (row[1] >= n) {
        n = row[1] + 1;
      }
    }

    const prob = new Problem();
    prob.m = m;
    prob.n = n;
    prob.nnz = data.length;
    prob.r = r;
    return prob;
  }
};
