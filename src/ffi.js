import koffi from 'koffi';
import path from 'node:path';
import url from 'node:url';

export const Node = koffi.struct('mf_node', {
  u: 'int',
  v: 'int',
  r: 'float'
});

export const Problem = koffi.struct('mf_problem', {
  m: 'int',
  n: 'int',
  nnz: 'long',
  r: 'mf_node *'
});

const Parameter = koffi.struct('mf_parameter', {
  fun: 'int',
  k: 'int',
  nr_threads: 'int',
  nr_bins: 'int',
  nr_iters: 'int',
  lambda_p1: 'float',
  lambda_p2: 'float',
  lambda_q1: 'float',
  lambda_q2: 'float',
  eta: 'float',
  alpha: 'float',
  c: 'float',
  do_nmf: 'bool',
  quiet: 'bool',
  copy_data: 'bool'
});

export const MfModel = koffi.struct('mf_model', {
  fun: 'int',
  m: 'int',
  n: 'int',
  k: 'int',
  b: 'float',
  p: 'float *',
  q: 'float *'
});

function defaultLib() {
  if (process.platform === 'win32') {
    return 'mf.dll';
  } else if (process.platform === 'darwin') {
    if (process.arch.startsWith('arm')) {
      return 'libmf.arm64.dylib';
    } else {
      return 'libmf.dylib';
    }
  } else {
    if (process.arch.startsWith('arm')) {
      return 'libmf.arm64.so';
    } else {
      return 'libmf.so';
    }
  }
}

const ffiLib = defaultLib();

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

const lib = koffi.load(path.join(__dirname, '..', 'vendor', ffiLib));

export default {
  mf_get_default_param: lib.func('mf_parameter mf_get_default_param()'),
  mf_read_problem: lib.func('mf_problem mf_read_problem(char *path)'),
  mf_save_model: lib.func('int mf_save_model(mf_model *model, char *path)'),
  mf_load_model: lib.func('mf_model* mf_load_model(char *path)'),
  mf_destroy_model: lib.func('void mf_destroy_model(mf_model **model)'),
  mf_train: lib.func('mf_model* mf_train(mf_problem *prob, mf_parameter param)'),
  mf_train_with_validation: lib.func('mf_model* mf_train_with_validation(mf_problem *tr, mf_problem *va, mf_parameter param)'),
  mf_cross_validation: lib.func('double mf_cross_validation(mf_problem *prob, int nr_folds, mf_parameter param)'),
  mf_predict: lib.func('float mf_predict(mf_model *model, int u, int v)'),
  calc_rmse: lib.func('double calc_rmse(mf_problem *prob, mf_model *model)'),
  calc_mae: lib.func('double calc_mae(mf_problem *prob, mf_model *model)'),
  calc_gkl: lib.func('double calc_gkl(mf_problem *prob, mf_model *model)'),
  calc_logloss: lib.func('double calc_logloss(mf_problem *prob, mf_model *model)'),
  calc_accuracy: lib.func('double calc_accuracy(mf_problem *prob, mf_model *model)'),
  calc_mpr: lib.func('double calc_mpr(mf_problem *prob, mf_model *model, bool transpose)'),
  calc_auc: lib.func('double calc_auc(mf_problem *prob, mf_model *model, bool transpose)')
};
