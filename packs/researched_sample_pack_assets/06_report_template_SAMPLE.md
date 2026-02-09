# Client Readout Template — Sample

## Executive Summary
- Stage: {{stage}} (confidence: {{confidence}})
- Top priorities:
  1) {{top1}}
  2) {{top2}}
  3) {{top3}}

## Stage Assessment
- Stage drivers:
  - {{driver_bullets}}
- Complexity factors:
  - {{complexity_bullets}}
- Unknowns affecting confidence:
  - {{unknowns}}

## Top Growth Areas (ranked)
{% for item in growth_areas %}
### {{loop.index}}) {{item.title}}
- Why now: {{item.why_now}}
- Evidence: {{item.evidence}}
- Impact: {{item.impact}}
- 30/60/90:
  - 30: {{item.plan30}}
  - 60: {{item.plan60}}
  - 90: {{item.plan90}}
- Metrics: {{item.metrics}}
- Follow-ups (if needed): {{item.followups}}
{% endfor %}

## Risk Register (not legal advice)
{% for risk in risks %}
- {{risk}}
{% endfor %}

## Disclaimer
This report is not legal advice. Consider reviewing compliance topics with counsel and/or your payroll/benefits advisor.
