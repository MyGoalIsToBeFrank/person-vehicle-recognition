#!/usr/bin/env node
/** 把 result.json 导出为每张图片一行、带原图缩略图的 Excel。 */

import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";
import sharp from "sharp";


const HERE = path.dirname(fileURLToPath(import.meta.url));
const bundledPython = path.join(HERE, "..", ".venv", "Scripts", "python.exe");
const python = existsSync(bundledPython) ? bundledPython : "python";
const config = JSON.parse(
  execFileSync(python, [path.join(HERE, "config.py")], { encoding: "utf8" }),
);

// 允许用命令行覆盖输入 JSON 和原图基目录，便于 Docker/CI 场景。
function parseCliArg(flag) {
  const token = process.argv.slice(2).find((arg) => arg.startsWith(`${flag}=`));
  return token ? token.slice(flag.length + 1) : undefined;
}
const INPUT_PATH = parseCliArg("--result-json") || config.RESULT_JSON;
const BASE_IMAGE_DIR = parseCliArg("--data-dir")
  ? path.resolve(parseCliArg("--data-dir"))
  : config.DATA_DIR;
const OUTPUT_PATH = parseCliArg("--output-xlsx")
  ? path.resolve(parseCliArg("--output-xlsx"))
  : config.RESULT_XLSX;
const THUMBNAIL_WIDTH_PX = 180;
const THUMBNAIL_HEIGHT_PX = 120;


function requireArray(value, location) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${location} 必须是数组`);
  }
  return value;
}


async function readResults() {
  const rows = JSON.parse(await fs.readFile(INPUT_PATH, "utf8"));
  requireArray(rows, "result.json 顶层");
  for (const [index, row] of rows.entries()) {
    const expectedKeys = ["图片位置", "处理耗时（毫秒）", "识别内容"];
    if (!row || expectedKeys.some((key) => !(key in row))) {
      throw new TypeError(`第 ${index + 1} 项缺少规定字段`);
    }
    requireArray(row["识别内容"]?.["行人"], `第 ${index + 1} 项的行人`);
    requireArray(row["识别内容"]?.["车辆"], `第 ${index + 1} 项的车辆`);
  }
  return rows;
}


function personText(person, index) {
  const styles = requireArray(person["上装"]?.["款式"], "上装款式").join("、") || "无";
  const lowers = requireArray(person["下装"], "下装").join("、") || "无";
  return (
    `${index + 1}. ${person["性别"]}｜${person["年龄"]}｜${person["朝向"]}`
    + `｜眼镜:${person["佩戴眼镜"]}｜帽子:${person["佩戴帽子"]}`
    + `｜手持:${person["手持物品"]}｜包:${person["包"]}`
    + `｜上装:${person["上装"]?.["袖长"] ?? ""}/${styles}`
    + `｜下装:${lowers}｜${person["鞋靴"]}｜口罩:${person["口罩"]}`
  );
}


function vehicleText(vehicle, index) {
  return `${index + 1}. ${vehicle["颜色"]} ${vehicle["车型"]}｜车牌:${vehicle["车牌"]}`;
}


function resultRows(results) {
  return results.map((result) => {
    const persons = result["识别内容"]["行人"];
    const vehicles = result["识别内容"]["车辆"];
    return [
      "",
      result["图片位置"],
      result["处理耗时（毫秒）"],
      persons.map(personText).join("\n") || "无",
      vehicles.map(vehicleText).join("\n") || "无",
    ];
  });
}


async function thumbnailData(imagePath) {
  const resolved = path.isAbsolute(imagePath)
    ? imagePath
    : path.join(BASE_IMAGE_DIR, imagePath);
  try {
    const { data, info } = await sharp(resolved)
      .rotate()
      .resize({
        width: THUMBNAIL_WIDTH_PX,
        height: THUMBNAIL_HEIGHT_PX,
        fit: "inside",
        withoutEnlargement: true,
      })
      .flatten({ background: "#FFFFFF" })
      .jpeg({ quality: 82 })
      .toBuffer({ resolveWithObject: true });
    return {
      dataUrl: `data:image/jpeg;base64,${data.toString("base64")}`,
      widthPx: info.width,
      heightPx: info.height,
    };
  } catch (error) {
    throw new Error(`无法读取导出图片: ${imagePath}`, { cause: error });
  }
}


async function buildWorkbook(results) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("识别结果");
  const headers = ["图像", "图片位置", "处理耗时（毫秒）", "行人", "车辆"];
  const rows = resultRows(results);
  const values = [headers, ...rows];
  const range = sheet.getRangeByIndexes(0, 0, values.length, headers.length);
  range.values = values;
  range.format.wrapText = true;
  range.format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#A6B8CE" },
  };

  const header = sheet.getRange("A1:E1");
  header.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  header.format.rowHeight = 26;

  if (rows.length) {
    const data = sheet.getRangeByIndexes(1, 0, rows.length, headers.length);
    data.format.verticalAlignment = "center";
    sheet.getRangeByIndexes(1, 2, rows.length, 1).format.numberFormat = "0.000";
  }
  [28, 62, 20, 72, 48].forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, values.length, 1).format.columnWidth = width;
  });

  const table = sheet.tables.add(`A1:E${values.length}`, true, "RecognitionResultsTable");
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;

  for (const [index, result] of results.entries()) {
    const persons = result["识别内容"]["行人"].length;
    const vehicles = result["识别内容"]["车辆"].length;
    const textLines = Math.max(persons, vehicles, 1);
    sheet.getRangeByIndexes(index + 1, 0, 1, headers.length).format.rowHeight = Math.max(
      96,
      Math.min(180, textLines * 18 + 12),
    );

    const thumbnail = await thumbnailData(result["图片位置"]);
    sheet.images.add({
      dataUrl: thumbnail.dataUrl,
      anchor: {
        from: { row: index + 1, col: 0 },
        extent: { widthPx: thumbnail.widthPx, heightPx: thumbnail.heightPx },
      },
    });
  }

  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  return workbook;
}


async function main() {
  const results = await readResults();
  const workbook = await buildWorkbook(results);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_PATH);
  // 当前表格运行时会在导出旁生成调试清单；它不是业务结果。
  await fs.rm(`${OUTPUT_PATH}.inspect.ndjson`, { force: true });
  console.log(`已导出 ${results.length} 张带缩略图的图片：${OUTPUT_PATH}`);
}


await main();
