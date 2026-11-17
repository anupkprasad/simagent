# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The user requires a single, comprehensive HTML report summarizing the analysis of eight 1 ns MD trajectories (four proteins, apo/holo). The provided JSONL file contains all computed metrics and image paths, so parsing it yields the necessary data for the report. To satisfy the literature‑correlation requirement, we perform PubMed searches for each protein focusing on activation‑loop dynamics and allosteric regulation. The resulting citations are passed to the report generator. The report will embed the analysis plots, display key statistics, and provide the literature context in a clean, professional layout.

## Overview

Generate a unified report (report.html) that: 1) Parses analysis_summary.jsonl; 2) Retrieves relevant PubMed references; 3) Uses `generate_html_report` to create an HTML file containing all visualizations, statistics, and literature commentary.

## Report Focus

- Cross‑comparison of apo vs. holo dynamics for ERBB3, VRK3, MLKL, and TITIN
- Correlation of activation‑loop RMSF and DSSP changes with ATP binding
- Allosteric coupling revealed by DCCM differences and COM‑distance analysis

## Execution Steps (6 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Load and parse the analysis results, statistics, and image file paths from analysis_summary.jsonl.

**Reason:** Need the analysis data to populate the report.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Search PubMed for ERBB3

**Tool:** `search_pubmed`

**Description:** Retrieve recent literature on ERBB3 activation‑loop dynamics and allosteric regulation.

**Reason:** Gather references for literature correlation.

**Parameters:**
```json
{
  "query": "ERBB3 activation loop dynamics",
  "max_results": 3,
  "include_abstracts": true,
  "max_age_years": null
}
```

### Step 3: Search PubMed for VRK3

**Tool:** `search_pubmed`

**Description:** Retrieve recent literature on VRK3 kinase regulation and activation‑loop behavior.

**Reason:** Gather references for literature correlation.

**Parameters:**
```json
{
  "query": "VRK3 kinase activation loop regulation",
  "max_results": 3,
  "include_abstracts": true,
  "max_age_years": null
}
```

### Step 4: Search PubMed for MLKL

**Tool:** `search_pubmed`

**Description:** Retrieve recent literature on MLKL allosteric regulation and activation‑loop dynamics.

**Reason:** Gather references for literature correlation.

**Parameters:**
```json
{
  "query": "MLKL allosteric regulation activation loop",
  "max_results": 3,
  "include_abstracts": true,
  "max_age_years": null
}
```

### Step 5: Search PubMed for TITIN

**Tool:** `search_pubmed`

**Description:** Retrieve recent literature on TITIN elastic domain dynamics and activation‑loop related movements.

**Reason:** Gather references for literature correlation.

**Parameters:**
```json
{
  "query": "Titin elastic domain dynamics",
  "max_results": 3,
  "include_abstracts": true,
  "max_age_years": null
}
```

### Step 6: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a comprehensive HTML report that embeds all analysis plots, displays key statistics, and includes the collected literature references.

**Reason:** Produce the final deliverable.

**Parameters:**
```json
{
  "analysis_data": "${{analysis_data}}",
  "literature_refs": "${{literature_refs}}",
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

