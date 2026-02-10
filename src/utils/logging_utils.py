"""Logging utilities for the replication system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Global logger registry
_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_string: Optional custom format string.

    Returns:
        Configured root logger.
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


class ExecutionLogger:
    """Logger for tracking code execution progress.

    This class provides structured logging for the replicator agent,
    capturing code execution details and timing information.
    """

    def __init__(self, paper_id: str, output_dir: str = "reports"):
        """Initialize execution logger.

        Args:
            paper_id: Identifier for the paper being replicated.
            output_dir: Directory for log output.
        """
        self.paper_id = paper_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_entries: list[dict] = []
        self.start_time = datetime.now()
        self.logger = get_logger(f"execution.{paper_id}")

    def log_step(
        self,
        step_name: str,
        status: str,
        details: Optional[dict] = None,
    ) -> None:
        """Log an execution step.

        Args:
            step_name: Name of the step.
            status: Status (started, completed, failed).
            details: Additional details.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "status": status,
            "details": details or {},
        }
        self.log_entries.append(entry)

        # Also log to standard logger
        self.logger.info(f"{step_name}: {status}")
        if details:
            self.logger.debug(f"Details: {details}")

    def log_code_execution(
        self,
        code_block: str,
        success: bool,
        output: str,
        error: Optional[str] = None,
    ) -> None:
        """Log a code execution result.

        Args:
            code_block: The code that was executed.
            success: Whether execution succeeded.
            output: Standard output.
            error: Error message if any.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "code_execution",
            "code_preview": code_block[:200] + "..." if len(code_block) > 200 else code_block,
            "success": success,
            "output_length": len(output),
            "error": error,
        }
        self.log_entries.append(entry)

        if success:
            self.logger.info("Code execution succeeded")
        else:
            self.logger.error(f"Code execution failed: {error}")

    def get_full_log(self) -> str:
        """Get the full execution log as a string.

        Returns:
            Formatted log string.
        """
        lines = [f"Execution Log for {self.paper_id}"]
        lines.append(f"Started: {self.start_time.isoformat()}")
        lines.append("=" * 50)

        for entry in self.log_entries:
            lines.append(f"\n[{entry['timestamp']}]")
            if "step" in entry:
                lines.append(f"  Step: {entry['step']}")
                lines.append(f"  Status: {entry['status']}")
            elif "type" in entry:
                lines.append(f"  Type: {entry['type']}")
                lines.append(f"  Success: {entry.get('success', 'N/A')}")
            if entry.get("error"):
                lines.append(f"  Error: {entry['error']}")

        return "\n".join(lines)

    def save_log(self) -> str:
        """Save the log to a file.

        Returns:
            Path to the saved log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.output_dir / f"execution_log_{self.paper_id}_{timestamp}.txt"

        with open(log_path, "w") as f:
            f.write(self.get_full_log())

        self.logger.info(f"Saved execution log to: {log_path}")
        return str(log_path)
