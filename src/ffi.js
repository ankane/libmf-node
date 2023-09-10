import ffi from 'ffi-napi';
import path from 'path';
import ref from 'ref-napi';
import refStructDi from 'ref-struct-di';
import url from 'url';

const StructType = refStructDi(ref);

export const Node = StructType({
  u: ref.types.int,
  v: ref.types.int,
  r: ref.types.float
});

export const Problem = StructType({
  m: ref.types.int,
  n: ref.types.int,
  nnz: ref.types.longlong,
  r: ref.refType(Node)
});

const Parameter = StructType({
  fun: ref.types.int,
  k: ref.types.int,
  nr_threads: ref.types.int,
  nr_bins: ref.types.int,
  nr_iters: ref.types.int,
  lambda_p1: ref.types.float,
  lambda_p2: ref.types.float,
  lambda_q1: ref.types.float,
  lambda_q2: ref.types.float,
  eta: ref.types.float,
  alpha: ref.types.float,
  c: ref.types.float,
  do_nmf: ref.types.bool,
  quiet: ref.types.bool,
  copy_data: ref.types.bool
});

const Model = StructType({
  fun: ref.types.int,
  m: ref.types.int,
  n: ref.types.int,
  k: ref.types.int,
  b: ref.types.float,
  p: ref.refType(ref.types.float),
  q: ref.refType(ref.types.float)
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

export default ffi.Library(path.join(__dirname, '..', 'vendor', ffiLib), {
  'mf_get_default_param': [Parameter, []],
  'mf_read_problem': [Problem, [ref.types.CString]],
  'mf_save_model': [ref.types.int, [ref.refType(Model), ref.types.CString]],
  'mf_load_model': [ref.refType(Model), [ref.types.CString]],
  'mf_destroy_model': [ref.types.void, [ref.refType(Model)]],
  'mf_train': [ref.refType(Model), [ref.refType(Problem), Parameter]],
  'mf_train_with_validation': [ref.refType(Model), [ref.refType(Problem), ref.refType(Problem), Parameter]],
  'mf_cross_validation': [ref.types.double, [ref.refType(Problem), ref.types.int, Parameter]],
  'mf_predict': [ref.types.float, [ref.refType(Model), ref.types.int, ref.types.int]],
  'calc_rmse': [ref.types.double, [ref.refType(Problem), ref.refType(Model)]],
  'calc_mae': [ref.types.double, [ref.refType(Problem), ref.refType(Model)]],
  'calc_gkl': [ref.types.double, [ref.refType(Problem), ref.refType(Model)]],
  'calc_logloss': [ref.types.double, [ref.refType(Problem), ref.refType(Model)]],
  'calc_accuracy': [ref.types.double, [ref.refType(Problem), ref.refType(Model)]],
  'calc_mpr': [ref.types.double, [ref.refType(Problem), ref.refType(Model), ref.types.bool]],
  'calc_auc': [ref.types.double, [ref.refType(Problem), ref.refType(Model), ref.types.bool]],
});
