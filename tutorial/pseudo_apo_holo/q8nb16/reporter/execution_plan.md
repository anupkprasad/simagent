# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The user wants a comprehensive HTML report for a single‑simulation analysis. The workflow requires parsing the pre‑generated analysis_summary.jsonl, extracting all relevant metrics and image paths, optionally enriching the content with literature that discusses activation‑loop dynamics and allosteric regulation for ERBB3, VRK3, MLKL, and TITIN, and finally generating a professional report that embeds the plots and key statistics. Because the user explicitly asked not to use the combined report tool, we will only generate one per‑simulation report and leave further merging to a higher‑level step if needed.

## Overview

1️⃣ Read the analysis summary. 2️⃣ Build literature queries tailored to each protein and the analysis types. 3️⃣ Retrieve top PubMed hits for each query. 4️⃣ Pass the parsed analysis data and literature references to `generate_html_report` to produce `report.html`. This keeps the workflow modular, reproducible, and easy to extend.

## Report Focus

- Activation‑loop conformations in ERBB3, VRK3, MLKL, TITIN
- Allosteric regulation revealed by DCCM and RMSF
- Comparison of apo vs holo dynamics
- MD/experimental dynamics context

## Execution Steps (4 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Parse the JSONL file containing all computed metrics, statistics, and image paths for the current system.

**Reason:** We need the raw data to feed into the report generator and to identify which literature searches are relevant.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate Literature Queries

**Tool:** `generate_literature_queries`

**Description:** Create PubMed search queries that combine protein aliases, family terms, and the specific analysis metrics (e.g., RMSD, RMSF, DSSP) to fetch papers describing activation‑loop conformations and allosteric regulation.

**Reason:** Targeted literature queries maximize the relevance of retrieved papers.

**Parameters:**
```json
{
  "analysis_types": [
    "RMSD",
    "RMSF",
    "DSSP",
    "C\u03b1 DCCM",
    "Radius of Gyration"
  ],
  "user_goal": "Compare backbone and loop dynamics across apo/holo states and relate to known allosteric mechanisms.",
  "protein_name": "Protein name inferred from the PDB ID in the directory name (e.g., ERBB3, VRK3, MLKL, TITIN)",
  "aliases": [],
  "gene_name": "",
  "uniprot_id": "",
  "analysis_stats": {}
}
```

### Step 3: Search PubMed for Relevant Literature

**Tool:** `search_pubmed`

**Description:** Execute each query from the previous step, retrieve up to 5 recent papers (last 10 years) per query, and collect titles, authors, abstracts, PubMed IDs, and DOI links.

**Reason:** We need actual literature references to embed in the report.

**Parameters:**
```json
{
  "query": "placeholder",
  "max_results": 5,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 4: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a comprehensive HTML report named `report.html` that embeds all analysis plots, presents key statistics, and lists the curated literature references.

**Reason:** This is the final deliverable the user requested.

**Parameters:**
```json
{
  "analysis_data": "placeholder",
  "literature_refs": "placeholder",
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

