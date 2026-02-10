"""Tests for logging utilities."""

import logging
from pathlib import Path

import pytest

from src.utils.logging_utils import ExecutionLogger, get_logger, setup_logging


class TestSetupLogging:
    def test_default_level(self):
        root = setup_logging(level="INFO")
        assert root.level == logging.INFO

    def test_debug_level(self):
        root = setup_logging(level="DEBUG")
        assert root.level == logging.DEBUG

    def test_with_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        logger = get_logger("test_file_logging")
        logger.info("Test message")
        assert log_file.exists()

    def test_custom_format(self):
        fmt = "%(levelname)s - %(message)s"
        root = setup_logging(format_string=fmt)
        assert any(
            h.formatter._fmt == fmt for h in root.handlers if h.formatter
        )


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_same_logger_returned(self):
        l1 = get_logger("same_name")
        l2 = get_logger("same_name")
        assert l1 is l2


class TestExecutionLogger:
    def test_log_step(self, tmp_path):
        el = ExecutionLogger("test_paper", str(tmp_path))
        el.log_step("setup", "started")
        el.log_step("setup", "completed")
        assert len(el.log_entries) == 2

    def test_log_code_execution(self, tmp_path):
        el = ExecutionLogger("test_paper", str(tmp_path))
        el.log_code_execution("print('hello')", True, "hello\n")
        assert len(el.log_entries) == 1
        assert el.log_entries[0]["success"] is True

    def test_log_code_execution_failure(self, tmp_path):
        el = ExecutionLogger("test_paper", str(tmp_path))
        el.log_code_execution("1/0", False, "", "ZeroDivisionError")
        assert el.log_entries[0]["success"] is False
        assert el.log_entries[0]["error"] == "ZeroDivisionError"

    def test_get_full_log(self, tmp_path):
        el = ExecutionLogger("test_paper", str(tmp_path))
        el.log_step("step1", "started")
        log = el.get_full_log()
        assert "test_paper" in log
        assert "step1" in log

    def test_save_log(self, tmp_path):
        el = ExecutionLogger("test_paper", str(tmp_path))
        el.log_step("step1", "done")
        log_path = el.save_log()
        assert Path(log_path).exists()
        content = Path(log_path).read_text()
        assert "test_paper" in content

    def test_long_code_truncated(self, tmp_path):
        el = ExecutionLogger("test_paper", str(tmp_path))
        long_code = "x = 1\n" * 100
        el.log_code_execution(long_code, True, "")
        assert el.log_entries[0]["code_preview"].endswith("...")
