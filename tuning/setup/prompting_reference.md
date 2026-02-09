# Prompting and Tuning Reference

This folder centralizes consultant-tunable model behavior.

## Files
- `../packs/default_pack.yaml`: stage bands, checks, weights, template path, prompt overrides.
- `../packs/researched_sample_pack.yaml`: alternate researched pack.
- `field_query_hints.default.json`: derived tracked fields + field-to-query hints for targeted retrieval wiring.

## System Prompt Behavior
The application uses three structured-output prompts from `app/llm/prompts.py`:

1. Snapshot Extraction
- Source-constrained extraction from chunks.
- Unknown-by-default behavior unless evidence is explicit.
- Requires evidence citations with `chunk_id` + snippet.

2. Headcount Fallback Inference
- Used only when direct headcount signals are missing.
- Must return low confidence and citations when inferred.

3. Plan Generation
- Uses only snapshot + findings.
- No new facts allowed.
- Requires action-level evidence citations.

## Prompt Overrides in Pack
Use `prompt_overrides` in the pack to customize:
- `snapshot_guidance`
- `plan_style`

## Practical Tuning Workflow
1. Clone `tuning/packs/default_pack.yaml` to a client-specific pack.
2. Edit checks, severity, and area weights.
3. Adjust `prompt_overrides` voice and strictness.
4. Run sample cases through the web app and compare output quality.
