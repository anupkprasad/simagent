# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The user wants a comprehensive HTML report that visualizes the MD trajectory analysis and places the results in a literature context. The workflow therefore requires parsing the existing JSONL results, optionally gathering relevant literature, and generating the report. The report generation tool automatically embeds the images referenced in the JSONL, so we only need to provide the parsed data and any literature references.

## Overview

1. Load the analysis_summary.jsonl file. 2. Generate PubMed queries based on the protein and analysis types. 3. Retrieve literature abstracts that discuss activation‑loop conformations, allostery, and MD dynamics for the target proteins. 4. Create a self‑contained HTML report that displays plots, statistics cards, and literature citations.

## Report Focus

- Backbone RMSD and COM distance evolution
- Per‑residue RMSF for residues 150–200
- Radius of gyration trends
- Dynamic cross‑correlation patterns
- DSSP secondary‑structure changes
- Comparative statistics across apo/holo systems

## Execution Steps (4 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Parse the JSONL file to obtain all quantitative results and image file paths.

**Reason:** We need the analysis data to feed into the report generator.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate Literature Queries

**Tool:** `generate_literature_queries`

**Description:** Formulate PubMed search strings that target activation‑loop dynamics and allostery for the proteins under study.

**Reason:** These queries will be fed into PubMed to fetch relevant literature.

**Parameters:**
```json
{
  "analysis_types": [
    "RMSD",
    "RMSF",
    "Cross\u2011Correlation",
    "DSSP"
  ],
  "user_goal": "Compare apo/holo dynamics of ERBB3, VRK3, MLKL, and TITIN and relate to known activation\u2011loop and allosteric regulation mechanisms.",
  "protein_name": "ERBB3, VRK3, MLKL, TITIN",
  "aliases": [],
  "gene_name": "",
  "uniprot_id": "",
  "analysis_stats": {}
}
```

### Step 3: Search PubMed

**Tool:** `search_pubmed`

**Description:** Retrieve abstracts and citations that match the generated queries.

**Reason:** Provide the report with up‑to‑date contextual literature.

**Parameters:**
```json
{
  "query": "<placeholder_for_generated_queries>",
  "max_results": 20,
  "include_abstracts": true,
  "max_age_years": null
}
```

### Step 4: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a professional, self‑contained report that embeds the plots, displays key statistics, and cites the literature.

**Reason:** Deliver the final user product.

**Parameters:**
```json
{
  "analysis_data": {},
  "literature_refs": {},
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

