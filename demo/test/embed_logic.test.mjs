import assert from "node:assert/strict";
import test from "node:test";
import { findCell, headerRow, clampTip, storyIsStale, evidenceIsStale, topPcs } from "../public/embed_logic.js";

const rows = [
  { site: "a", parent: "p1", pc: "PC10", explained: "False" },
  { site: "a", parent: "p2", pc: "PC10", explained: "True" },
  { site: "b", parent: "p1", pc: "PC10", explained: "True" },
];

test("findCell is per site and parent", () => {
  assert.equal(findCell(rows, "a", "p2", "PC10").explained, "True");
  assert.equal(findCell(rows, "a", "p1", "PC10").explained, "False");
  assert.equal(findCell(rows, "b", "p9", "PC10"), undefined);
});

test("headerRow prefers an explained row", () => {
  assert.equal(headerRow(rows, "PC10").explained, "True");
});

test("clampTip flips and stays in viewport", () => {
  assert.deepEqual(clampTip(100, 100, 340, 80, 1000, 800), { left: 114, top: 114 });
  const r = clampTip(900, 780, 340, 80, 1000, 800);
  assert.equal(r.left, 900 - 14 - 340);
  assert.equal(r.top, 780 - 14 - 80);
  assert.equal(clampTip(5, 5, 2000, 2000, 300, 300).left, 8);
});

test("storyIsStale compares recorded call", () => {
  assert.equal(storyIsStale({ call: "Batch_3" }, "Batch_3"), false);
  assert.equal(storyIsStale({ call: "Batch_3" }, "Batch_2"), true);
  assert.equal(storyIsStale(undefined, "Batch_2"), true);
});

test("evidenceIsStale compares the recorded call", () => {
  assert.equal(evidenceIsStale({ a: "Batch_1" }, "a", "Batch_1"), false);
  assert.equal(evidenceIsStale({ a: "Batch_1" }, "a", "Batch_2"), true);
  assert.equal(evidenceIsStale({}, "a", "Batch_1"), true);
});

test("topPcs ranks over the displayed sites only", () => {
  const r = [
    { site: "a", pc: "PC1", contribution: "1" }, { site: "a", pc: "PC2", contribution: "-3" },
    { site: "hidden", pc: "PC1", contribution: "50" },
  ];
  assert.deepEqual(topPcs(r, ["a"], 2), ["PC2", "PC1"]);
  assert.deepEqual(topPcs(r, ["a"], 1), ["PC2"]);
});

test("reopenSite keeps the overlay only for a displayed site on the calls view", async () => {
  const { reopenSite } = await import("../public/embed_logic.js");
  assert.equal(reopenSite("a", "embeddings", ["a", "b"]), "a");
  assert.equal(reopenSite("a", "embeddings", ["b"]), null);
  assert.equal(reopenSite("a", "proof", ["a"]), null);
  assert.equal(reopenSite(null, "embeddings", ["a"]), null);
});
