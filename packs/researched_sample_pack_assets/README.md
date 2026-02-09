# Researched Sample Package (for testing)

This package includes:
- A filled stage framework + rubric + check library.
- A YAML consultant pack example (packs/researched_sample_pack.yaml).
- Two fictional company input packs under `inputs/researched_sample_pack/`.
- Two “reference” example outputs under `tests/golden/researched_sample_pack/expected_outputs/` (handwritten, for comparison).
- Gold cases (answer keys) under `tests/golden/researched_sample_pack/gold_cases/`.

## How to use
1) Point your tool at `packs/researched_sample_pack.yaml`
2) Run ingestion on one of:
   - `inputs/researched_sample_pack/northstar_analytics/`
   - `inputs/researched_sample_pack/brick_brew/`
3) Compare your output to:
   - `tests/golden/researched_sample_pack/gold_cases/GC-001.md` (Northstar)
   - `tests/golden/researched_sample_pack/gold_cases/GC-002.md` (Brick & Brew)
   - `tests/golden/researched_sample_pack/expected_outputs/*.md` for a narrative reference

## Notes
- Compliance items are prompts only; not legal advice.
- Sources informing thresholds are listed in SOURCES.md.
