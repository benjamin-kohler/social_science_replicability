"""Abstract base class for all replication runners."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..models.schemas import PaperSummary
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts


class BaseReplicationRunner(ABC):
    """Abstract base class for all replication runners.

    All runners share a common ``run()`` signature so that
    ``BenchmarkRunner`` can dispatch uniformly.
    """

    def __init__(self, timeout: int = 600, allow_web_access: bool = False,
                 item_types: list[str] | None = None):
        self.timeout = timeout
        self.allow_web_access = allow_web_access
        self.item_types = item_types or ["table", "figure"]

    @abstractmethod
    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
        paper_direct: bool = False,
    ) -> RunArtifacts:
        """Run a replication and return the workspace artifacts.

        Args:
            model: Model specification.
            paper: Paper specification with paths.
            paper_summary: Pre-extracted methodology summary.
            workspace_dir: Directory where the runner should produce output.
            paper_direct: If True, provide the raw paper PDF to the replicator.

        Returns:
            RunArtifacts with workspace_dir, stdout/stderr, exit code, etc.
        """
        ...
