# Knowledge Base Directory

This directory contains domain knowledge for the planner agent to reference during execution planning.

## Structure

### `/md_fundamentals`
Research papers and documentation on molecular dynamics fundamentals:
- MD theory and methods
- Sampling techniques
- Statistical mechanics principles
- Simulation best practices

### `/force_fields`
Information about force fields:
- AMBER99SB-ILDN parameters and usage
- CHARMM force fields
- OPLS-AA documentation
- Small molecule parameterization guides

### `/protocols`
Standard MD protocols and workflows:
- Protein preparation protocols
- Membrane protein setup
- Ligand binding simulations
- Free energy calculations
- Enhanced sampling methods

### `/tools_manuals`
Software tool manuals and command references:
- GROMACS commands and usage
- VMD scripting guides
- Analysis tool documentation
- AmberTools manuals

## Usage

The planner **retrieves** top-k chunks (hashed n-gram + Jaccard, optional
Ollama embeddings) instead of dumping whole files. Citations look like
`[kb:protocols.equilibration_protocol#nvt-temperature]`. The index is
persisted at `planner/knowledge_index.json` (next to this tree, and copied
into the campaign `planner/` folder). The repository copy is generated locally
and is not published.

Documents can be in:
- Markdown (.md)
- Plain text (.txt)
- JSON (.json) for structured data
- PDF (.pdf) for papers and manuals

## Adding Knowledge

1. Place documents in the appropriate subdirectory
2. Use descriptive filenames (e.g., `amber99sb_parameters.md`)
3. Keep documents focused and well-organized
4. The planner will automatically discover and load new files

## Examples

```
knowledge/
├── md_fundamentals/
│   └── md_theory_basics.md
├── force_fields/
│   ├── amber99sb_ildn.md
│   └── charmm36.md
├── protocols/
│   ├── protein_preparation_workflow.md
│   ├── equilibration_protocol.md
│   └── ligand_binding_simulations.md
└── tools_manuals/
    └── gromacs_pdb2gmx.md
```

## Notes

- The planner has a context limit, so keep documents concise
- Focus on actionable information and practical guidance
- Cross-reference related documents when helpful
- Update this README when adding new categories
