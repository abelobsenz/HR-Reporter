from __future__ import annotations

from pathlib import Path
import logging

from jinja2 import Environment, FileSystemLoader

from app.models import FinalReport

logger = logging.getLogger(__name__)

def render_markdown(
    report: FinalReport,
    profile_name: str = "Default Profile",
    repo_root: str | Path | None = None,
) -> str:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    template_path = root / "app" / "report" / "templates" / "report.md.j2"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    template = env.get_template(template_path.name)
    rendered = template.render(
        report=report.model_dump(mode="python"),
        profile_name=profile_name,
    )
    logger.info("report_rendered template=%s chars=%s", template_path, len(rendered))
    return rendered
