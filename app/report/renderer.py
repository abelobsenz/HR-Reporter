from __future__ import annotations

from pathlib import Path
import logging

from jinja2 import Environment, FileSystemLoader

from app.models import ConsultantPack, FinalReport

logger = logging.getLogger(__name__)

def render_markdown(report: FinalReport, pack: ConsultantPack, repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    template_path = Path(pack.templates.markdown_template)
    if not template_path.is_absolute():
        template_path = root / template_path

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    template = env.get_template(template_path.name)
    rendered = template.render(report=report.model_dump(mode="python"), pack=pack.model_dump(mode="python"))
    logger.info("report_rendered template=%s chars=%s", template_path, len(rendered))
    return rendered
