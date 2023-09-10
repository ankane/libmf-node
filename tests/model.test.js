import { Loss, Matrix, Model } from 'libmf';
import fs from 'fs';

test('works', () => {
  const data = readFile('real_matrix.tr.txt');

  let model = new Model({quiet: true});
  model.fit(data);

  expect(model.rows()).toBe(2309);
  expect(model.columns()).toBe(1368);
  expect(model.factors()).toBe(8);
  model.bias();
  expect(model.p().length).toBe(model.rows());
  expect(model.p()[0].length).toBe(model.factors());
  expect(model.q().length).toBe(model.columns());
  expect(model.q()[0].length).toBe(model.factors());

  const pred = model.predict(1, 1);
  const path = '/tmp/model.txt';
  model.save(path);
  model = Model.load(path);
  expect(model.predict(1, 1)).toBe(pred);

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
  expect(model.rows()).toBe(2309);
});

test('path eval set', () => {
  const model = new Model({quiet: true});
  model.fit(filePath('real_matrix.tr.txt'), filePath('real_matrix.te.txt'));
  expect(model.rows()).toBe(2309);
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
  expect(() => model.bias()).toThrow('Not fit');
});

test('no data', () => {
  const model = new Model();
  expect(() => model.fit(new Matrix())).toThrow('No data');
});

test('save missing', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true});
  model.fit(data);
  expect(() => model.save('missing/model.txt')).toThrow('Cannot save model');
});

test('load missing', () => {
  expect(() => Model.load('missing.txt')).toThrow('Cannot open model');
});

test('fit bad param', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true, factors: 0});
  expect(() => model.fit(data)).toThrow('fit failed');
});

test('cv bad param', () => {
  const data = readFile('real_matrix.tr.txt');
  const model = new Model({quiet: true, factors: 0});
  expect(() => model.cv(data)).toThrow('cv failed');
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
