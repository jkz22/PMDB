export const MAD_TO_SD = 1.4826;
export const MIN_MAD_TO_SD_RATIO = 0.1;
export const MAX_Z = 10.0;
export const SCALE_SHRINK = 0.5;
export const DEFAULT_OOD_ALPHA = 0.1;

export const FEATURES = [
  "si_depth_rel_band0",
  "si_depth_rel_band1",
  "si_depth_rel_band2",
  "si_depth_rel_band3",
  "si_depth_rel_band4",
  "si_depth_slope",
  "si_depth_mid_dip",
  "gx_0.5_2.0",
  "gx_2.0_4.0",
  "gx_4.0_7.0",
  "gx_7.0_10.0",
  "gz_0.5_2.0",
  "gz_2.0_4.0",
  "gz_4.0_7.0",
  "gz_7.0_10.0",
  "k15_contact_tilestd",
];

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return Number.NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function columnValues(matrix, column) {
  return matrix.map((row) => row[column]);
}

function nanMedianColumns(matrix) {
  return Array.from({ length: matrix[0].length }, (_, j) => median(columnValues(matrix, j)));
}

function nanStdColumns(matrix) {
  return Array.from({ length: matrix[0].length }, (_, j) => {
    const values = columnValues(matrix, j).filter(Number.isFinite);
    if (values.length === 0) return Number.NaN;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  });
}

function robustScale(residuals) {
  const medAbs = nanMedianColumns(residuals.map((row) => row.map(Math.abs)));
  const mad = medAbs.map((value) => MAD_TO_SD * value);
  const sd = nanStdColumns(residuals);
  return mad.map((value, j) => (value >= MIN_MAD_TO_SD_RATIO * sd[j] ? value : sd[j]));
}

function batchParams(z, codes, nBatches) {
  const means = [];
  const scales = [];
  for (let batch = 0; batch < nBatches; batch += 1) {
    const batchRows = z.filter((_, index) => codes[index] === batch);
    const center = nanMedianColumns(batchRows);
    const rawScale = robustScale(batchRows.map((row) => row.map((value, j) => value - center[j])));
    means.push(center);
    scales.push(rawScale.map((value) => {
      const shrunk = SCALE_SHRINK + (1 - SCALE_SHRINK) * value;
      return shrunk > 0 ? shrunk : 1;
    }));
  }
  return [means, scales];
}

function scores(z, means, scales) {
  return z.map((row) => means.map((center, batch) => {
    let sum = 0;
    for (let j = 0; j < row.length; j += 1) {
      const scale = scales[batch][j];
      sum += Math.abs(row[j] - center[j]) / scale + Math.log(scale);
    }
    return sum / row.length;
  }));
}

function fitCore(x, codes, nBatches) {
  const center = nanMedianColumns(x);
  const scale = robustScale(x.map((row) => row.map((value, j) => value - center[j])));
  const keep = scale.map((value) => value > 0);
  if (!keep.some(Boolean)) throw new Error("all features have zero spread");
  const keptCenter = center.filter((_, j) => keep[j]);
  const keptScale = scale.filter((_, j) => keep[j]);
  const z = x.map((row) => row
    .filter((_, j) => keep[j])
    .map((value, j) => Math.max(-MAX_Z, Math.min(MAX_Z, (value - keptCenter[j]) / keptScale[j]))));
  const [means, scales] = batchParams(z, codes, nBatches);
  return { center, scale, keep, means, scales };
}

function matrixFromRows(rows, features) {
  return rows.map((row, index) => features.map((feature) => {
    const value = row[feature];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`incomplete fingerprint at row ${index}: ${feature}`);
    }
    return value;
  }));
}

export function fit(rows, labels, features) {
  if (rows.length !== labels.length) throw new Error("rows and labels must have the same length");
  const requested = [...features];
  const batches = [...new Set(labels.map(String))].sort();
  const codes = labels.map((label) => batches.indexOf(String(label)));
  const x = matrixFromRows(rows, requested);
  const { center, scale, keep, means, scales } = fitCore(x, codes, batches.length);
  const keptFeatures = requested.filter((_, j) => keep[j]);
  const keptX = x.map((row) => row.filter((_, j) => keep[j]));
  return {
    features: keptFeatures,
    batches,
    center: center.filter((_, j) => keep[j]),
    scale: scale.filter((_, j) => keep[j]),
    batchCenter: means,
    batchScale: scales,
    trainX: keptX,
    trainCodes: codes,
  };
}

function looScores(z) {
  const result = [];
  for (let i = 0; i < z.length; i += 1) {
    const others = z.filter((_, index) => index !== i);
    const center = nanMedianColumns(others);
    const rawScale = robustScale(others.map((row) => row.map((value, j) => value - center[j])));
    const scale = rawScale.map((value) => {
      const shrunk = SCALE_SHRINK + (1 - SCALE_SHRINK) * value;
      return shrunk > 0 ? shrunk : 1;
    });
    result.push(scores([z[i]], [center], [scale])[0][0]);
  }
  return result;
}

function fullConformalP(trainX, trainCodes, nBatches, xNew) {
  const augmented = [...trainX, xNew];
  const center = nanMedianColumns(augmented);
  const scale = robustScale(augmented.map((row) => row.map((value, j) => value - center[j])));
  const z = augmented.map((row) => row.map((value, j) => Math.max(
    -MAX_Z,
    Math.min(MAX_Z, (value - center[j]) / scale[j]),
  )));
  const zTrain = z.slice(0, -1);
  const zNew = z.at(-1);
  const p = [];
  for (let batch = 0; batch < nBatches; batch += 1) {
    const batchRows = zTrain.filter((_, index) => trainCodes[index] === batch);
    const loo = looScores([...batchRows, zNew]);
    const last = loo.at(-1);
    p.push(loo.filter((score) => score >= last).length / loo.length);
  }
  return p;
}

function nonconformityScores(model, rows) {
  const x = matrixFromRows(rows, model.features);
  const z = x.map((row) => row.map((value, j) => Math.max(
    -MAX_Z,
    Math.min(MAX_Z, (value - model.center[j]) / model.scale[j]),
  )));
  return scores(z, model.batchCenter, model.batchScale);
}

function argmin(values) {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] < values[best]) best = i;
  }
  return best;
}

export function predict(model, rows, { oodAlpha = DEFAULT_OOD_ALPHA } = {}) {
  const allScores = nonconformityScores(model, rows);
  const counts = model.batches.map((_, batch) => model.trainCodes.filter((code) => code === batch).length);
  return rows.map((row, index) => {
    const p = fullConformalP(
      model.trainX,
      model.trainCodes,
      model.batches.length,
      matrixFromRows([row], model.features)[0],
    );
    const winner = argmin(allScores[index]);
    const rejected = p.map((value, batch) => (
      value < oodAlpha || value <= 1 / (counts[batch] + 1) + 1e-12
    ));
    const others = p.filter((_, batch) => batch !== winner);
    return {
      assigned: model.batches[winner],
      credibility: p[winner],
      confidence: 1 - Math.max(...others),
      ood: rejected.every(Boolean),
      p: Object.fromEntries(model.batches.map((batch, j) => [batch, p[j]])),
      score: Object.fromEntries(model.batches.map((batch, j) => [batch, allScores[index][j]])),
    };
  });
}

export function looAssignments(rows, labels, features) {
  const batches = [...new Set(labels.map(String))].sort();
  const codes = labels.map((label) => batches.indexOf(String(label)));
  const x = matrixFromRows(rows, features);
  const assigned = [];
  for (let i = 0; i < x.length; i += 1) {
    const trainX = x.filter((_, index) => index !== i);
    const trainCodes = codes.filter((_, index) => index !== i);
    const { center, scale, keep, means, scales } = fitCore(trainX, trainCodes, batches.length);
    const z = x[i]
      .filter((_, j) => keep[j])
      .map((value, j) => Math.max(-MAX_Z, Math.min(MAX_Z, (value - center.filter((_, k) => keep[k])[j])
        / scale.filter((_, k) => keep[k])[j])));
    assigned.push(batches[argmin(scores([z], means, scales)[0])]);
  }
  return assigned;
}

export function drift(row, t) {
  const result = { ...row };
  result.si_depth_rel_band0 *= 1 - 0.5 * t;
  result.si_depth_rel_band1 *= 1 - 0.3 * t;
  result.si_depth_rel_band3 *= 1 + 0.4 * t;
  result.si_depth_rel_band4 *= 1 + 0.8 * t;
  result.si_depth_slope = result.si_depth_rel_band4 - result.si_depth_rel_band0;
  result.si_depth_mid_dip = result.si_depth_rel_band2
    - (result.si_depth_rel_band0 + result.si_depth_rel_band4) / 2;
  for (const feature of Object.keys(result)) {
    if (feature.startsWith("gx_") || feature.startsWith("gz_")) {
      result[feature] *= 1 - 0.35 * t;
    }
  }
  result.k15_contact_tilestd *= 1 + 4 * t;
  return result;
}
