export default class Matrix {
  constructor() {
    this.data = [];
  }

  push(rowIndex, columnIndex, value) {
    this.data.push({u: rowIndex, v: columnIndex, r: value});
  }
};
