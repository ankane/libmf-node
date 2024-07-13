import ffi, { MfModel, Node, Problem } from './ffi.js';
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
    return ffi.mf_predict(this.#model(), rowIndex, columnIndex);
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
    if (model === null) {
      throw new Error('Cannot open model');
    }
    this.#setModel(model);
  }

  rows() {
    return this.#decodedModel().m;
  }

  columns() {
    return this.#decodedModel().n;
  }

  factors() {
    return this.#decodedModel().k;
  }

  bias() {
    return this.#decodedModel().b;
  }

  p() {
    return this.#readFactors(this.#decodedModel().p, this.rows());
  }

  q() {
    return this.#readFactors(this.#decodedModel().q, this.columns());
  }

  rmse(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_rmse(prob, this.#model());
  }

  mae(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_mae(prob, this.#model());
  }

  gkl(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_gkl(prob, this.#model());
  }

  logloss(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_logloss(prob, this.#model());
  }

  accuracy(data) {
    const prob = this.#createProblem(data);
    return ffi.calc_accuracy(prob, this.#model());
  }

  mpr(data, transpose) {
    const prob = this.#createProblem(data);
    return ffi.calc_mpr(prob, this.#model(), transpose);
  }

  auc(data, transpose) {
    const prob = this.#createProblem(data);
    return ffi.calc_auc(prob, this.#model(), transpose);
  }

  // TODO do this automatically
  destroy() {
    this.#destroyModel();
  }

  // TODO improve performance
  #readFactors(ptr, n) {
    const f = this.factors();
    const factors = [];
    let offset = 0;
    for (let i = 0; i < n; i++) {
      factors.push(koffi.decode(ptr, offset, koffi.types.float, f));
      offset += 4 * f;
    }
    return factors;
  }

  #model() {
    if (!this.model) {
      throw new Error('Not fit');
    }
    return this.model;
  }

  #decodedModel() {
    return koffi.decode(this.#model(), MfModel);
  }

  #setModel(model) {
    this.#destroyModel();
    this.model = model;
  }

  #destroyModel() {
    if (this.model) {
      ffi.mf_destroy_model([this.model]);
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
    const r = Buffer.allocUnsafe(koffi.sizeof(Node) * data.length);
    koffi.encode(r, Node, data, data.length);
    let offset = 0;

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
    prob.r = r;

    const buf = Buffer.allocUnsafe(koffi.sizeof(Problem));
    koffi.encode(buf, Problem, prob);
    return buf;
  }
};
