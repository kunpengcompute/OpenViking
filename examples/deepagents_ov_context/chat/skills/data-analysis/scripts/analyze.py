# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Inspect and query local tables without runtime installers or shared caches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def load_tables(files: list[Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in files:
        if path.suffix.lower() == ".xlsx":
            frames = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        elif path.suffix.lower() in {".csv", ".tsv"}:
            frames = {
                path.stem: pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
            }
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")
        for name, frame in frames.items():
            base = re.sub(r"[^\w]", "_", str(name)) or "table"
            if base[0].isdigit():
                base = "t_" + base
            unique, index = base, 2
            while unique in tables:
                unique, index = f"{base}_{index}", index + 1
            tables[unique] = frame
    return tables


def analyze(files: list[Path], action: str, sql: str | None = None) -> dict[str, Any]:
    tables = load_tables(files)
    if action == "query":
        if not sql:
            raise ValueError("query requires --sql")
        with duckdb.connect() as connection:
            statements = connection.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise ValueError("Use one SELECT query")
            for name, frame in tables.items():
                connection.register(name, frame)
            frame = connection.execute(sql).fetchdf()
            return {
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "records": json.loads(frame.to_json(orient="records", date_format="iso")),
            }
    result: dict[str, Any] = {}
    for name, frame in tables.items():
        if action == "inspect":
            result[name] = {
                "rows": len(frame),
                "columns": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
                "nulls": {str(column): int(value) for column, value in frame.isna().sum().items()},
                "sample": json.loads(frame.head(5).to_json(orient="records", date_format="iso")),
            }
        elif action == "summary":
            result[name] = json.loads(frame.describe(include="all").to_json(date_format="iso"))
        else:
            raise ValueError("Unknown action")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--action", choices=("inspect", "query", "summary"), required=True)
    parser.add_argument("--sql")
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    result = analyze(args.files, args.action, args.sql)
    if args.output_file:
        if args.action != "query" or args.output_file.suffix not in {".csv", ".json"}:
            parser.error("Export requires a query and .csv or .json output")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        if args.output_file.suffix == ".csv":
            pd.DataFrame(result["records"], columns=result["columns"]).to_csv(
                args.output_file, index=False, encoding="utf-8-sig"
            )
        else:
            args.output_file.write_text(
                json.dumps(result["records"], ensure_ascii=False, allow_nan=False)
            )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
