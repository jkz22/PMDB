export function parseCSV(text) {
  const records = [];
  let record = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ",") {
      record.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      record.push(field);
      records.push(record);
      record = [];
      field = "";
      if (char === "\r" && text[i + 1] === "\n") i += 1;
    } else {
      field += char;
    }
  }

  if (field.length > 0 || record.length > 0) {
    record.push(field);
    records.push(record);
  }

  while (records.length > 0 && records.at(-1).length === 1 && records.at(-1)[0] === "") {
    records.pop();
  }
  if (records.length === 0) return [];

  const headers = records.shift();
  const values = (raw) => {
    if (raw === "") return null;
    const trimmed = raw.trim();
    if (trimmed !== "" && /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(trimmed)) {
      const number = Number(trimmed);
      if (Number.isFinite(number)) return number;
    }
    return raw;
  };

  return records
    .filter((row) => !(row.length === 1 && row[0] === ""))
    .map((row) => Object.fromEntries(headers.map((header, index) => [
      header,
      values(row[index] ?? ""),
    ])));
}
