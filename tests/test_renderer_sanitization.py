from __future__ import annotations

from app.models import FinalReport, GuidanceReference, StageBasedRecommendation
from app.report.renderer import render_markdown


def test_guidance_sources_hide_non_http_urls() -> None:
    report = FinalReport(
        stage="50-99 employees",
        stage_based_recommendation=StageBasedRecommendation(
            size_stage="50-99 employees",
            funding_stage="Series B",
            hr_structure_recommendation="People lead with specialist support.",
            sources=[
                GuidanceReference(title="Internal template", url="/Users/secret/template.docx"),
                GuidanceReference(title="External article", url="https://example.com/hr-guide"),
            ],
        ),
    )

    rendered = render_markdown(report)

    assert "/Users/" not in rendered
    assert "Internal template (internal)" in rendered
    assert "https://example.com/hr-guide" in rendered
