import assert from 'node:assert';
import test from 'node:test';
import { Loss, Matrix, Model } from 'libmf';
import fs from 'node:fs';

test('works', () => {
  const data = readFile('real_matrix.tr.txt');

  let model = new Model({quiet: true});
  model.fit(data);

  assert.equal(model.rows(), 2309);
  assert.equal(model.columns(), 1368);
  assert.equal(model.factors(), 8);
  model.bias();
  assert.equal(model.p().length, model.rows());
  assert.equal(model.p()[0].length, model.factors());
  assert.equal(model.q().length, model.columns());
  assert.equal(model.q()[0].length, model.factors());

  const pred = model.predict(1, 1);
  const path = '/tmp/model.txt';
  model.save(path);
  model = Model.load(path);
  assert.equal(model.predict(1, 1), pred);

  // test destroy twice
  model.destroy();
  model.destroy();
});

test('eval set', () => {
  const trainSet = readFile('real_matrix.tr.txt');
  const evalSet = readFile('real_matrix.te.txt');

  const model = new Model({quiet: true});
  model.fit(trainSet, evalSet);
  model.rmse(evalSet);
});

test('eval set extra', () => {
  const trainSet = readFile('real_matrix.tr.txt');
  const evalSet = new Matrix();
  evalSet.push(1000000, 1000000, 1);

  const model = new Model({quiet: true});
  model.fit(trainSet, evalSet);
  assert.equal(model.rows(), 2309);
  assert.equal(model.columns(), 1368);
});

test('eval set extra ONE_CLASS_L2', () => {
  const trainSet = readFile('real_matrix.tr.txt');
  const evalSet = new Matrix();
  evalSet.push(1000000, 1000000, 1);

  const model = new Model({loss: Loss.ONE_CLASS_L2});
  assert.throws(() => model.fit(trainSet, evalSet), {message: 'Extra indices in eval set not supported for ONE_CLASS_L2 loss'});
});

test('path', () => {
  const model = new Model({quiet: true});
  assert.throws(() => model.fit(filePath('real_matrix.tr.txt')), {message: 'Reading data directly from files is no longer supported'});
});

test('cv', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true});
  model.cv(data);
});

test('negative row index', () => {
  const data = new Matrix();
  data.push(-1, 0, 1);
  const model = new Model({quiet: true});
  assert.throws(() => model.fit(data), {message: 'Invalid row index'});
});

test('max row index', () => {
  const data = new Matrix();
  data.push(2**31 - 1, 0, 1);
  const model = new Model({quiet: true});
  assert.throws(() => model.fit(data), {message: 'Invalid row index'});
});

test('negative column index', () => {
  const data = new Matrix();
  data.push(0, -1, 1);
  const model = new Model({quiet: true});
  assert.throws(() => model.fit(data), {message: 'Invalid column index'});
});

test('max column index', () => {
  const data = new Matrix();
  data.push(0, 2**31 - 1, 1);
  const model = new Model({quiet: true});
  assert.throws(() => model.fit(data), {message: 'Invalid column index'});
});

test('loss real_kl', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true, loss: Loss.REAL_KL});
  model.fit(data);
});

test('not fit', () => {
  const model = new Model({quiet: true});
  assert.throws(() => model.bias(), {message: 'Not fit'});
});

test('no data', () => {
  const model = new Model();
  assert.throws(() => model.fit(new Matrix()), {message: 'No data'});
});

test('save missing', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true});
  model.fit(data);
  assert.throws(() => model.save('missing/model.txt'), {message: 'Cannot save model'});
});

test('load missing', () => {
  assert.throws(() => Model.load('missing.txt'), {message: 'Cannot open model'});
});

test('fit bad param', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true, factors: 0});
  assert.throws(() => model.fit(data), {message: 'fit failed'});
});

test('cv bad param', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true, factors: 0});
  assert.throws(() => model.cv(data), {message: 'cv failed'});
});

function filePath(filename) {
  return `vendor/demo/${filename}`;
}

function readFile(filename) {
  const data = new Matrix();
  const lines = fs.readFileSync(filePath(filename), 'utf-8').split('\n');
  lines.pop();
  for (let line of lines) {
    const row = line.split(' ');
    data.push(parseInt(row[0]), parseInt(row[1]), parseFloat(row[2]));
  }
  return data;
}
