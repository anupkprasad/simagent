# Planner Execution Plan

**Generated:** 2026-11-13 20:34:01
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** reporter_agent
**Subtask:** reporter_only

## Full Plan

**Execution Plan for Generating the Scientific Report**

---

### 1. Goal Clarification  
The user wants a comprehensive, publication‑ready report that:

1. **Summarises the pre‑computed metrics** (backbone RMSD, per‑residue RMSF, radius of gyration, COM distance for holo systems, Cα DCCM, DCCM differences, DSSP evolution) for all eight trajectories (apo/holo for ERBB3, VRK3, MLKL, TITIN).  
2. **Presents comparative overlay plots and statistical tables** that juxtapose these metrics across the four proteins and their apo/holo states.  
3. **Integrates literature** on activation‑loop conformations, allosteric regulation, and MD/experimental dynamics for each protein, and discusses how the simulation data align with or diverge from the literature.  
4. **Outputs a polished HTML report** in the designated `reports/` directory, with embedded figures, tables, and a narrative discussion.

No new trajectory calculations are required; the report will be built solely from the already‑generated analysis artefacts.

---

### 2. Data Acquisition and Parsing  

#### 2.1 Locate and Read Analysis Summaries  
- **Tool**: `read_analysis_summary`  
- **Action**: For each of the eight simulation directories under `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42_ATP_MG/analysis/`, invoke the tool with the path to its `analysis_summary.jsonl`.  
- **Outcome**: A structured Python dictionary (or list of dictionaries) containing all metric values, file paths to plots, and metadata for that simulation.  
- **Why**: This is the authoritative source of all pre‑computed metrics; parsing it once per simulation guarantees consistency.

#### 2.2 Verify Presence of Required Artefacts  
- **Tool**: `list_dir` (optional) and `grep_file` (if needed)  
- **Action**: Confirm that each summary references the expected plot files (e.g., RMSD time‑series, RMSF bar plots, DCCM heatmaps, DSSP evolution curves).  
- **Outcome**: A checklist of available artefacts; any missing items will be flagged for inclusion in the final report as “data not available”.

---

### 3. Comparative Visualisation and Statistical Summaries  

#### 3.1 Generate Combined Overlay Report  
- **Tool**: `generate_combined_html_report`  
- **Inputs**:  
  - `sim_dirs`: list of the eight analysis directories.  
  - `labels`: human‑readable labels such as “ERBB3‑apo”, “ERBB3‑holo”, etc.  
  - `overlay_plots`: the tool will automatically locate overlay‑ready plots (e.g., RMSD, RMSF, Rg, COM distance, DCCM differences, DSSP).  
  - `working_dir`: the root workspace (`/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42_ATP_MG`).  
- **Action**: The tool will read each `analysis_summary.jsonl`, embed the corresponding plots, and produce a self‑contained HTML page that juxtaposes the metrics across all systems.  
- **Outcome**: A ready‑to‑publish HTML file containing side‑by‑side plots, a statistical table (mean, SD, min, max for each metric), and a brief caption for each figure.

#### 3.2 Extract Numerical Tables  
- **Tool**: `read_analysis_summary` (again)  
- **Action**: Pull out the numeric values for each metric and assemble them into a CSV or Markdown table.  
- **Outcome**: A clean table that can be embedded in the final report and used for quick reference.

---

### 4. Literature Retrieval and Contextualisation  

#### 4.1 UniProt Functional Annotations  
- **Tool**: `search_uniprot`  
- **Queries**: “ERBB3”, “VRK3”, “MLKL”, “TITIN”  
- **Action**: Retrieve reviewed entries, gene aliases, functional descriptions, and any curated literature references.  
- **Outcome**: A concise protein‑specific background section for each protein.

#### 4.2 PubMed Search Queries  
- **Tool**: `generate_literature_queries`  
- **Analysis Types**:  
  - Activation‑loop dynamics  
  - Allosteric regulation mechanisms  
  - MD or experimental dynamics studies  
- **Action**: Generate a set of PubMed queries tailored to each protein and each theme.  
- **Outcome**: A list of query strings ready for execution.

#### 4.3 PubMed Retrieval  
- **Tool**: `search_pubmed`  
- **Action**: For each query string, fetch up to 20–30 recent (last 10 years) articles, including titles, abstracts, and citation counts.  
- **Outcome**: A curated bibliography for each protein, highlighting key findings relevant to the simulation metrics.

---

### 5. Final Report Assembly  

#### 5.1 Combine Analysis and Literature  
- **Tool**: `generate_html_report`  
- **Inputs**:  
  - `analysis_data`: the aggregated dictionary of metrics and plot paths from step 2.  
  - `literature_refs`: the list of PubMed article metadata from step 4.  
- **Action**:  
  - Create a structured HTML document with the following sections:  
    1. **Abstract** – concise summary of objectives, methods, key findings.  
    2. **Introduction** – background on the four proteins, relevance of ATP binding, and the importance of activation‑loop dynamics.  
    3. **Methods** – brief description of the simulation setup (already documented in the analysis artefacts) and the metrics computed.  
    4. **Results** –  
       - Sub‑section for each protein, presenting the comparative overlay plots (from step 3) and the statistical tables.  
       - Highlight specific observations (e.g., holo systems show reduced RMSF in residues 150–200, COM distance trends).  
    5. **Discussion** –  
       - Compare simulation observations with literature findings (activation‑loop conformations, allosteric effects).  
       - Discuss any discrepancies or confirmations, and propose mechanistic interpretations.  
    6. **Conclusion** – summarize the main insights and suggest future work.  
    7. **References** – formatted bibliography from step 4.  
  - Embed all figures and tables directly into the HTML for portability.  
- **Outcome**: A single, self‑contained HTML file ready for dissemination or conversion to PDF.

#### 5.2 Save Report  
- **Location**: `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42_ATP_MG/reports/`  
- **File Name**: `ERBB3_VRK3_MLKL_TITIN_MD_Report.html` (or similar).  

---

### 6. Quality Assurance and Final Checks  

1. **Verify that all eight systems are represented** in the final report; if any summary was missing, note it in a “Data Availability” note.  
2. **Confirm that all referenced plots exist** and load correctly in the HTML.  
3. **Cross‑check numeric tables** against the original `analysis_summary.jsonl` entries to ensure no transcription errors.  
4. **Ensure literature citations** are properly formatted and linked (if possible).  
5. **Run a quick HTML preview** in a browser to catch any rendering issues.

---

### 7. Summary of Agents and Tools Used  

| Agent | Purpose | Tool(s) |
|-------|---------|---------|
| Reporter | Orchestrates the entire workflow | `read_analysis_summary`, `generate_combined_html_report`, `search_uniprot`, `generate_literature_queries`, `search_pubmed`, `generate_html_report` |
| Reporter | Optional sanity checks | `list_dir`, `grep_file` |

No analysis or programming agents are invoked, in strict compliance with the user’s instruction that all metrics are pre‑computed.

---

By following this plan, the Reporter Agent will produce a comprehensive, publication‑ready HTML report that synthesises the eight pre‑computed MD analyses, visualises comparative metrics, and contextualises the findings within the current scientific literature on ERBB3, VRK3, MLKL, and TITIN.