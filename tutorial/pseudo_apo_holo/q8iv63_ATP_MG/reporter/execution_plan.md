# Reporter Execution Plan

**Generated:** N/A

## LLM Reasoning

The per‑simulation report must incorporate the quantitative results that were already computed and stored in the `analysis_summary.jsonl` file, and it should contextualise these findings with recent literature on the protein’s activation loop and allosteric regulation.  Therefore the workflow consists of (1) loading the analysis data, (2) preparing a set of PubMed queries that target the most relevant literature, (3) (optionally) retrieving the literature via `search_pubmed`, and finally (4) generating a self‑contained HTML report that embeds the analysis plots and highlights key statistics.  No combined‑simulation steps are required because the user wants a single‑simulation report.

The `generate_html_report` tool will automatically parse the JSONL, extract the required statistics and image paths, embed the images as base64, and create the modern layout.  The only thing we must supply explicitly is the file name (`report.html`) and the type (`comprehensive`).

We also include a lightweight literature‑query generation step with `generate_literature_queries` so that the report can pull in up‑to‑date references relevant to each protein.  These queries can be refined manually if desired, but the tool will produce a sensible set based on the analysis types and protein aliases.

All file paths are relative to the reporter’s working directory; therefore we use the relative filename `analysis_summary.jsonl` for reading and `report.html` for writing.

The overall complexity is moderate because the heavy lifting (plot creation, data aggregation) has already been done by the analysis agent.



## Overview

1. Load analysis data from `analysis_summary.jsonl`. 2. Generate literature queries for the protein of interest. 3. (Optional) Run PubMed searches to retrieve abstracts and citations. 4. Create a comprehensive HTML report (`report.html`) that embeds the analysis visualisations, highlights key statistics, and summarises the literature context.

The report is automatically self‑contained and ready for review.



## Report Focus

- Activation‑loop conformational dynamics
- Allosteric regulation and ATP‑binding effects

## Execution Steps (4 steps)

### Step 1: Read Analysis Summary

**Tool:** `read_analysis_summary`

**Description:** Parse the JSONL file containing RMSD, RMSF, gyration, DCCM, DSSP, and image paths.

**Reason:** Load all pre‑computed analysis results into a structured dictionary for the report generator.

**Parameters:**
```json
{
  "summary_file": "analysis_summary.jsonl"
}
```

### Step 2: Generate Literature Queries

**Tool:** `generate_literature_queries`

**Description:** Create PubMed search queries that combine protein aliases, dynamics, and activation‑loop themes.

**Reason:** Provide structured search strings that will retrieve relevant literature from PubMed.

**Parameters:**
```json
{
  "analysis_types": [
    "RMSD",
    "RMSF",
    "Radius of Gyration",
    "DCCM",
    "DSSP"
  ],
  "user_goal": "Analyze MD trajectories of protein activation loops and correlate with experimental dynamics",
  "protein_name": "ERBB3",
  "aliases": [
    "p21860",
    "ERBB3",
    "HER3",
    "ErbB3"
  ],
  "gene_name": "ERBB3",
  "uniprot_id": "P21860",
  "analysis_stats": {}
}
```

### Step 3: Search PubMed

**Tool:** `search_pubmed`

**Description:** Retrieve a short set of recent papers for each query to include in the report.

**Reason:** Gather up‑to‑date literature that can be embedded as citations in the HTML report.

**Parameters:**
```json
{
  "query": "ERBB3 activation loop dynamics",
  "max_results": 5,
  "include_abstracts": true,
  "max_age_years": 10
}
```

### Step 4: Generate HTML Report

**Tool:** `generate_html_report`

**Description:** Create a comprehensive report that embeds all analysis plots, statistics, and literature references.

**Reason:** Produce the final deliverable that satisfies the user’s requirement for a professional, self‑contained HTML document.

**Parameters:**
```json
{
  "analysis_data": {},
  "literature_refs": [],
  "report_type": "comprehensive",
  "output_file": "report.html"
}
```

