# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The goal is to transform the raw MD‑analysis output (stored in `analysis/analysis_summary.jsonl`) into a polished, data‑rich HTML report.  The report must embed all generated plots, present key statistics, and contextualise the findings with recent literature on activation‑loop conformations and allosteric regulation for each of the four pseudokinases.  The workflow therefore consists of: (1) loading the analysis data, (2) creating PubMed search queries that capture dynamics/structure‑function literature for the four proteins, (3) querying PubMed and assembling a concise reference list, and finally (4) feeding the analysis results and the literature references into the `generate_html_report` tool to create `report.html` in the reporter directory.

## Overview

1. Parse `analysis_summary.jsonl`. 2. Generate literature queries for ERBB3, VRK3, MLKL, and TITIN. 3. Retrieve recent papers (last 10 years) that discuss activation‑loop dynamics, allosteric regulation, or MD‑derived insights. 4. Construct a single‑simulation HTML report that embeds plots, displays statistics cards, and lists key references.

## Report Focus

- Backbone RMSD and stability trends
- Residue‑level flexibility (RMSF, especially 150–200)
- Global compactness (radius of gyration)
- Ligand‑binding pocket dynamics (COM distance)
- Cooperative motions (DCCM differences)
- Secondary‑structure evolution (DSSP)

## Execution Steps (7 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Parse the JSONL file containing all analysis metrics, statistics, and image file paths.

**Reason:** We need the analysis data and image paths to populate the report.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate Literature Search Queries

**Tool:** `generate_literature_queries`

**Description:** Create PubMed search queries tailored to the four pseudokinases and the analysis types (RMSD, RMSF, DSSP, DCCM, allosteric regulation).

**Reason:** These queries will drive targeted PubMed searches that return relevant recent literature.

**Parameters:**
```json
{
  "analysis_types": [
    "RMSD",
    "RMSF",
    "DSSP",
    "DCCM",
    "allosteric",
    "activation-loop"
  ],
  "user_goal": "Analyze activation-loop dynamics and allosteric regulation in ERBB3, VRK3, MLKL, TITIN from MD trajectories.",
  "protein_name": "",
  "aliases": [
    "ERBB3",
    "VRK3",
    "MLKL",
    "TITIN"
  ],
  "gene_name": "",
  "uniprot_id": "",
  "analysis_stats": {}
}
```

### Step 3: Search PubMed for ERBB3

**Tool:** `search_pubmed`

**Description:** Retrieve the most recent papers on ERBB3 activation‑loop dynamics and allosteric regulation.

**Reason:** Collect a focused reference list for ERBB3.

**Parameters:**
```json
{
  "query": "ERBB3 activation loop dynamics allosteric regulation",
  "max_results": 10,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 4: Search PubMed for VRK3

**Tool:** `search_pubmed`

**Description:** Retrieve the most recent papers on VRK3 pseudokinase dynamics and regulation.

**Reason:** Collect a focused reference list for VRK3.

**Parameters:**
```json
{
  "query": "VRK3 pseudokinase dynamics allosteric regulation",
  "max_results": 10,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 5: Search PubMed for MLKL

**Tool:** `search_pubmed`

**Description:** Retrieve the most recent papers on MLKL activation‑loop dynamics and necroptosis regulation.

**Reason:** Collect a focused reference list for MLKL.

**Parameters:**
```json
{
  "query": "MLKL activation loop dynamics necroptosis allosteric",
  "max_results": 10,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 6: Search PubMed for TITIN

**Tool:** `search_pubmed`

**Description:** Retrieve the most recent papers on TITIN structural dynamics and allosteric regulation in muscle.

**Reason:** Collect a focused reference list for TITIN.

**Parameters:**
```json
{
  "query": "TITIN structural dynamics allosteric regulation",
  "max_results": 10,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 7: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a comprehensive, self‑contained HTML report that embeds all analysis plots, displays key statistics cards, and lists the curated literature references.

**Reason:** Produce the final user deliverable.

**Parameters:**
```json
{
  "analysis_data": {},
  "literature_refs": [],
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

