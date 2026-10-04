import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import test from "node:test";

import { parseCSV } from "../lib/csv.js";
import * as fingerprint from "../lib/fingerprint.js";
import * as confound from "../lib/confound.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixtures = JSON.parse(readFileSync(path.join(root, "demo/test/fixtures.json"), "utf8"));
const readCsv = (relativePath) => parseCSV(readFileSync(path.join(root, relativePath), "utf8"));

function close(actual, expected, context = "value") {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", `${context}: expected a number`);
    assert.ok(
      Math.abs(actual - expected) <= 1e-9,
      `${context}: expected ${expected}, received ${actual}`,
    );
    return;
  }
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${context}: expected an array`);
    assert.equal(actual.length, expected.length, `${context}: array length`);
    expected.forEach((value, index) => close(actual[index], value, `${context}[${index}]`));
    return;
  }
  if (expected && typeof expected === "object") {
    assert.ok(actual && typeof actual === "object", `${context}: expected an object`);
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${context}: keys`);
    for (const key of Object.keys(expected)) close(actual[key], expected[key], `${context}.${key}`);
    return;
  }
  assert.equal(actual, expected, context);
}

function trainingInput() {
  const rows = readCsv("outputs/fingerprint/features.csv");
  return {
    records: rows,
    features: rows.map((row) => Object.fromEntries(
      fixtures.fingerprint.features.map((feature) => [feature, row[feature]]),
    )),
    labels: rows.map((row) => row.batch),
  };
}

function expectedFromPredictionCsv(row) {
  return {
    batch: row.batch,
    site: row.site,
    assigned: row.assigned,
    credibility: row.credibility,
    confidence: row.confidence,
    ood: row.ood === "True",
    p: Object.fromEntries(fingerprintModelBatches.map((batch) => [batch, row[`p_${batch}`]])),
    score: Object.fromEntries(fingerprintModelBatches.map((batch) => [batch, row[`score_${batch}`]])),
  };
}

function verifyPrediction(actual, expected, context) {
  assert.equal(actual.assigned, expected.assigned, `${context}: assigned`);
  assert.equal(actual.ood, expected.ood, `${context}: ood`);
  const winner = actual.score
    ? Object.keys(actual.score).reduce((best, batch) => (
      actual.score[batch] < actual.score[best] ? batch : best
    ))
    : null;
  assert.equal(actual.assigned, winner, `${context}: argmin(score)`);
  close(actual.credibility, expected.credibility, `${context}.credibility`);
  close(actual.confidence, expected.confidence, `${context}.confidence`);
  close(actual.p, expected.p, `${context}.p`);
  close(actual.score, expected.score, `${context}.score`);
}

const fingerprintModelBatches = ["Batch_1", "Batch_2", "Batch_3"];

test("parseCSV handles quoting, CRLF, numeric fields, empty cells, and boolean strings", () => {
  const parsed = parseCSV(
    'name,number,empty,flag,note\r\nalpha,1.25,,True,"say ""hi"",\r\nthere"\r\n',
  );
  assert.deepEqual(parsed, [{
    name: "alpha",
    number: 1.25,
    empty: null,
    flag: "True",
    note: 'say "hi",\r\nthere',
  }]);
});

test("fingerprint fit matches Python model and held-out predictions", () => {
  const { records, features: rows, labels } = trainingInput();
  const inputs = fixtures.fingerprint.inputs.trainRows;
  assert.equal(records.length, inputs.length);
  const recordsByKey = new Map(records.map((row) => [`${row.batch}/${row.site}`, row]));
  for (const reference of inputs) {
    const csvRow = recordsByKey.get(`${reference.batch}/${reference.site}`);
    assert.ok(csvRow, `missing training row ${reference.batch}/${reference.site}`);
    for (const feature of fixtures.fingerprint.features) {
      close(csvRow[feature], reference[feature], `${reference.site}.${feature} CSV input`);
    }
  }

  const model = fingerprint.fit(rows, labels, fixtures.fingerprint.features);
  close(model, fixtures.fingerprint.model, "fitted model");
  assert.equal(fixtures.featuresMatch, true);
  close(fixtures.maxFeatureDiff, 0, "feature table difference");

  const heldoutCsv = readCsv("outputs/fingerprint/heldout_features.csv");
  assert.equal(heldoutCsv.length, fixtures.fingerprint.inputs.heldoutRows.length);
  const heldoutRows = heldoutCsv.map((row) => Object.fromEntries(
    model.features.map((feature) => [feature, row[feature]]),
  ));
  const predictions = fingerprint.predict(model, heldoutRows);
  const expectedPredictions = fixtures.fingerprint.heldout.predictions;
  predictions.forEach((prediction, index) => {
    verifyPrediction(prediction, expectedPredictions[index], `heldout ${index}`);
  });

  const storedCsv = readCsv("outputs/fingerprint/heldout_predictions.csv");
  const storedByKey = new Map(storedCsv.map((row) => [`${row.batch}/${row.site}`, row]));
  expectedPredictions.forEach((expected) => {
    const stored = storedByKey.get(`${expected.batch}/${expected.site}`);
    assert.ok(stored, `missing stored prediction ${expected.site}`);
    verifyPrediction(expectedFromPredictionCsv(stored), expected, `stored ${expected.site}`);
  });
});

test("fingerprint drift grid, assignments, and accuracy match Python", () => {
  const { features: rows, labels } = trainingInput();
  const model = fingerprint.fit(rows, labels, fixtures.fingerprint.features);
  const basePrediction = fingerprint.predict(model, rows);
  const batch3Indices = labels.flatMap((label, index) => label === "Batch_3" ? [index] : []);
  const baseIndex = batch3Indices.reduce((best, index) => (
    basePrediction[index].p.Batch_3 > basePrediction[best].p.Batch_3 ? index : best
  ), batch3Indices[0]);
  const base = rows[baseIndex];
  assert.equal(`${labels[baseIndex]}/${trainingInput().records[baseIndex].site}`,
    `${fixtures.fingerprint.drift.base.batch}/${fixtures.fingerprint.drift.base.site}`);
  close(base, fixtures.fingerprint.drift.base.features, "drift base features");

  for (const expected of fixtures.fingerprint.drift.grid) {
    const drifted = fingerprint.drift(base, expected.t);
    close(drifted, expected.features, `drift features t=${expected.t}`);
    const [prediction] = fingerprint.predict(model, [drifted]);
    verifyPrediction(prediction, expected.prediction, `drift t=${expected.t}`);
  }

  const assignments = fingerprint.looAssignments(rows, labels, fixtures.fingerprint.features);
  assert.deepEqual(assignments, fixtures.fingerprint.loo.assignments.map((row) => row.assigned));
  const correct = assignments.filter((assigned, index) => assigned === labels[index]).length;
  assert.equal(correct, fixtures.fingerprint.loo.correct);
  assert.equal(correct, 21);
  assert.equal(rows.length, 31);
});

test("confound LOO and offset sweeps match Python", () => {
  const stats = readCsv("outputs/raw_intensity_stats.csv")
    .filter((row) => row.detector === "BSE")
    .map((row) => ({
      batch: row.batch,
      site: row.site,
      p1: row.p1,
      p50: row.p50,
      frac_zero: row.frac_zero,
    }));
  const referenceRows = fixtures.confound.rows;
  assert.equal(stats.length, referenceRows.length);
  const byKey = new Map(referenceRows.map((row) => [`${row.batch}/${row.site}`, row]));
  for (const row of stats) {
    const expected = byKey.get(`${row.batch}/${row.site}`);
    assert.ok(expected, `missing confound row ${row.site}`);
    close(
      Object.fromEntries(confound.FEATURES.map((feature) => [feature, row[feature]])),
      Object.fromEntries(confound.FEATURES.map((feature) => [feature, expected[feature]])),
      `confound input ${row.site}`,
    );
  }

  const rows = stats.map((row) => confound.FEATURES.map((feature) => row[feature]));
  const labels = stats.map((row) => row.batch);
  const accuracy = confound.looAccuracy(rows, labels);
  close(accuracy, fixtures.confound.looAccuracy, "confound LOO accuracy");
  assert.equal(Math.round(accuracy * 100) / 100, 0.71);

  for (const site of fixtures.confound.sweeps) {
    const rowIndex = stats.findIndex((row) => row.batch === site.batch && row.site === site.site);
    assert.ok(rowIndex >= 0, `missing confound site ${site.batch}/${site.site}`);
    const source = rows[rowIndex];
    for (const expected of site.predictions) {
      const offset = confound.offsetSite(source, expected.delta);
      close(offset, expected.features, `${site.site} offset ${expected.delta}`);
      assert.equal(
        confound.nearestCentroid(rows, labels, offset),
        expected.assigned,
        `${site.site} assignment at ${expected.delta}`,
      );
    }
  }
});
