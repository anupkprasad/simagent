# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The task requires creating a single‑simulation HTML report from pre‑generated analysis results. No new simulations are needed, so the workflow simply parses the provided JSONL summary, then feeds that data into the report generator. Optional literature searches are omitted to keep the plan focused on the required deliverables, but the structure can be expanded if needed.

## Overview

1️⃣ Load analysis results from `analysis_summary.jsonl`. 2️⃣ Generate a comprehensive HTML report (`report.html`) that embeds all plots, displays key statistics, and follows the modern layout guidelines.

## Report Focus

- Backbone RMSD
- Per‑residue RMSF (residues 150–200)
- Radius of Gyration
- ATP‑Catalytic Pocket COM Distance (holo only)
- Cα DCCM
- DSSP Evolution (whole protein & residues 150–200)

## Execution Steps (2 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Parse the JSONL file that contains all analysis metrics, statistics, and image paths.

**Reason:** To obtain the data that will drive the report.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a self‑contained, visually polished HTML report that embeds all analysis visualisations and key statistics.

**Reason:** Produce the final deliverable.

**Parameters:**
```json
{
  "analysis_data": {},
  "output_file": "report.html",
  "report_type": "comprehensive"
}
```

