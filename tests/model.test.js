import assert from 'node:assert';
import test from 'node:test';
import { Loss, Matrix, Model } from 'libmf';
import fs from 'fs';

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

test('path', () => {
  const model = new Model({quiet: true});
  model.fit(filePath('real_matrix.tr.txt'));
  assert.equal(model.rows(), 2309);
});

test('path eval set', () => {
  const model = new Model({quiet: true});
  model.fit(filePath('real_matrix.tr.txt'), filePath('real_matrix.te.txt'));
  assert.equal(model.rows(), 2309);
});

test('cv', () => {
  const data = filePath('real_matrix.tr.txt');
  const model = new Model({quiet: true});
  model.cv(data);
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
