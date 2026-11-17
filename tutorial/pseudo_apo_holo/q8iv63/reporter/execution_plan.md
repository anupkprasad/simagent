# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The report must summarize the MD analyses for each of the four proteins (p21860, q8iv63, q8nb16, q8wz42) and embed all visualizations.  The analysis results are stored in a single `analysis_summary.jsonl` file, which the `read_analysis_summary` tool can parse.  The parsed data (including image file paths and statistics) will be fed into `generate_html_report` to produce a comprehensive, self‑contained HTML document with base64‑encoded images, key statistics cards, and a modern layout.  To add literature context, we first generate PubMed search queries with `generate_literature_queries`, then retrieve the top results with `search_pubmed`.  The literature references will be incorporated into the final report via the `literature_refs` parameter of `generate_html_report`.

## Overview

1. Parse the analysis summary. 2. Create literature search queries per protein. 3. Fetch literature citations. 4. Generate a single HTML report (`report.html`) that embeds all plots, statistics, and literature.

## Report Focus

- Comparison of apo vs holo dynamics across four proteins
- Correlation of simulation metrics with known activation‑loop conformations and allosteric mechanisms

## Execution Steps (4 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Load the JSONL file that contains all analysis results, statistics, and image paths.

**Reason:** We need the raw analysis data to feed into the report generator.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate Literature Queries

**Tool:** `generate_literature_queries`

**Description:** Produce PubMed search queries for each protein to retrieve recent studies on activation‑loop conformation, allosteric regulation, and MD dynamics.

**Reason:** We need tailored PubMed queries for each protein and analysis metric.

**Parameters:**
```json
{
  "analysis_types": [
    "RMSD",
    "RMSF",
    "DCCM",
    "DSSP",
    "COM distance"
  ],
  "user_goal": "Analyze the eight existing 1\u2011ns trajectories (apo and holo for p21860, q8iv63, q8nb16, q8wz42) to compute, for each system, (1) backbone RMSD over time, (2) per\u2011residue RMSF with a bar plot for residues\u202f150\u2013200 where they exist, (3) radius of gyration, and (4) for holo systems only, the COM distance between ATP and the catalytic pocket defined as protein atoms within 5\u202f\u00c5 of ATP in frame\u202f0. Generate per\u2011protein C\u03b1 DCCMs and, where both apo and holo cases are present, compute the DCCM difference; also produce DSSP time series for the entire protein and the 150\u2013200 window. Compile overlay plots and statistical tables that compare these metrics across all four proteins. In the final report, retrieve literature on each protein\u2019s activation\u2011loop conformation, allosteric regulation, and MD or experimental dynamics, and correlate those findings with the simulation analyses.",
  "protein_name": "",
  "aliases": [
    "p21860",
    "q8iv63",
    "q8nb16",
    "q8wz42"
  ],
  "gene_name": "",
  "uniprot_id": "",
  "analysis_stats": {}
}
```

### Step 3: Search PubMed for Each Query

**Tool:** `search_pubmed`

**Description:** Iterate over the queries generated above and fetch the top literature results to include in the report.

**Reason:** Collect relevant, recent literature for context.

**Parameters:**
```json
{
  "query": "<placeholder_query>",
  "max_results": 5,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 4: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a comprehensive HTML file that embeds all analyses, visualizations, statistics, and literature references.

**Reason:** Produce the final deliverable as a self‑contained, readable report.

**Parameters:**
```json
{
  "analysis_data": "<analysis_summary_data>",
  "literature_refs": "<merged_literature_results>",
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

