"""Agent 0: Collector - Organizes papers and data for the replication pipeline."""

import shutil
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel

from ..models.schemas import PaperEntry
from ..models.config import Config
from ..utils.logging_utils import get_logger
from .base import BaseAgent

logger = get_logger(__name__)


class CollectorAgent(BaseAgent):
    """Agent 0: Collects and organizes paper inputs.

    This agent takes a list of PaperEntry objects and organizes them into
    a standard directory structure under ``data/input/<paper_id>/``.
    """

    def __init__(self, config: Config, chat_model: Optional[BaseChatModel] = None):
        super().__init__(
            config=config,
            name="Collector",
            role="paper collection organizer",
            goal="Organize papers and data into a standard directory structure",
            chat_model=chat_model,
        )

    def run(
        self,
        papers: list[PaperEntry],
        base_dir: str = "data/input",
    ) -> dict[str, Path]:
        """Organize papers into the standard directory structure.

        For each paper, creates ``base_dir/<paper_id>/`` containing:
        - ``paper.pdf`` (copied from ``entry.pdf_path``)
        - ``data/`` subdirectory with all data files
        - ``replication_package/`` if provided

        Args:
            papers: List of PaperEntry objects to organize.
            base_dir: Root directory for organized inputs.

        Returns:
            Mapping of paper_id to its organized directory path.
        """
        logger.info(f"Collecting {len(papers)} paper(s) into {base_dir}")
        result = {}

        for entry in papers:
            paper_dir = Path(base_dir) / entry.paper_id
            paper_dir.mkdir(parents=True, exist_ok=True)

            # Copy PDF
            pdf_src = Path(entry.pdf_path)
            if pdf_src.exists():
                shutil.copy2(pdf_src, paper_dir / "paper.pdf")
            else:
                logger.warning(f"[{entry.paper_id}] PDF not found: {entry.pdf_path}")

            # Copy data files
            data_dir = paper_dir / "data"
            data_dir.mkdir(exist_ok=True)
            for dp in entry.data_paths:
                dp_path = Path(dp)
                if dp_path.exists():
                    if dp_path.is_dir():
                        dest = data_dir / dp_path.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(dp_path, dest)
                    else:
                        shutil.copy2(dp_path, data_dir / dp_path.name)
                else:
                    logger.warning(f"[{entry.paper_id}] Data file not found: {dp}")

            # Copy replication package if provided
            if entry.replication_package_path:
                rp_path = Path(entry.replication_package_path)
                if rp_path.exists():
                    rp_dest = paper_dir / "replication_package"
                    if rp_dest.exists():
                        shutil.rmtree(rp_dest)
                    if rp_path.is_dir():
                        shutil.copytree(rp_path, rp_dest)
                    else:
                        rp_dest.mkdir(exist_ok=True)
                        shutil.copy2(rp_path, rp_dest / rp_path.name)
                else:
                    logger.warning(
                        f"[{entry.paper_id}] Replication package not found: "
                        f"{entry.replication_package_path}"
                    )

            result[entry.paper_id] = paper_dir
            logger.info(f"[{entry.paper_id}] Organized into {paper_dir}")

        return result
