export const FEATURES = ["p1", "p50", "frac_zero"];
export const BATCHES = ["Batch_1", "Batch_2", "Batch_3"];

function values(row) {
  return Array.isArray(row) ? row : FEATURES.map((feature) => row[feature]);
}

function sampleStd(rows, column) {
  const columnValues = rows.map((row) => values(row)[column]);
  const mean = columnValues.reduce((sum, value) => sum + value, 0) / columnValues.length;
  const squared = columnValues.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  return Math.sqrt(squared / (columnValues.length - 1));
}

export function nearestCentroid(trainRows, trainLabels, x) {
  const train = trainRows.map(values);
  const query = values(x);
  const std = FEATURES.map((_, column) => sampleStd(trainRows, column));
  const distances = BATCHES.map((batch) => {
    const rows = train.filter((_, index) => trainLabels[index] === batch);
    const center = FEATURES.map((_, column) => (
      rows.reduce((sum, row) => sum + row[column], 0) / rows.length
    ));
    return Math.sqrt(center.reduce((sum, value, column) => (
      sum + ((query[column] - value) / std[column]) ** 2
    ), 0));
  });
  let best = 0;
  for (let index = 1; index < distances.length; index += 1) {
    if (distances[index] < distances[best]) best = index;
  }
  return BATCHES[best];
}

export function looAccuracy(rows, labels) {
  let correct = 0;
  for (let i = 0; i < rows.length; i += 1) {
    const trainRows = rows.filter((_, index) => index !== i);
    const trainLabels = labels.filter((_, index) => index !== i);
    if (nearestCentroid(trainRows, trainLabels, rows[i]) === labels[i]) correct += 1;
  }
  return correct / rows.length;
}

export function offsetSite(x, delta) {
  const [p1, p50, fracZero] = values(x);
  return [p1 + delta, p50 + delta, delta > 0 ? 0 : fracZero];
}
