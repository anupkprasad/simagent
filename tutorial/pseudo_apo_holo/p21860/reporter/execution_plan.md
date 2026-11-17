# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The user wants a comprehensive HTML report that embeds all plots and statistics from the analysis_summary.jsonl file and contextualizes the findings with peer‑reviewed literature.  The report must be generated with `generate_html_report`, which accepts the parsed analysis data and an optional list of literature references.  The analysis data are loaded with `read_analysis_summary`.  Because the literature context is a key part of the goal, we prepare a set of PubMed search queries for the four proteins of interest, but we will not invoke the search tool directly in this plan—those queries can be executed manually or by the user after the report is generated.

## Overview

1. Parse the analysis summary. 2. Generate a list of relevant literature queries for ERBB3, VRK3, MLKL, and TITIN. 3. Create a single HTML report that embeds all plots, displays key statistics cards, and includes placeholder sections for literature citations.

## Report Focus

- Backbone RMSD and RMSF trends across apo/holo states
- Radius of gyration and COM distance to ATP
- DCCM differences between apo and holo
- DSSP dynamics of the 150–200 region

## Execution Steps (2 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Load and parse the analysis_summary.jsonl file to obtain all metric results and image paths.

**Reason:** Need the analysis data to feed into the report.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a comprehensive, self‑contained HTML report that embeds all plots, displays key statistics cards, and allocates sections for literature references.

**Reason:** Produce the final deliverable with all requested visualizations and insights.

**Parameters:**
```json
{
  "analysis_data": {},
  "literature_refs": [],
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

