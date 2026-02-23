"""Tests for the Collector agent."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.collector import CollectorAgent
from src.models.schemas import PaperEntry


def _mock_chat_model():
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="")
    return mock


class TestCollectorAgent:
    def test_init(self, config):
        agent = CollectorAgent(config, chat_model=_mock_chat_model())
        assert agent.name == "Collector"

    def test_run_organizes_files(self, config, tmp_path):
        # Create source files
        pdf_src = tmp_path / "source" / "paper.pdf"
        pdf_src.parent.mkdir()
        pdf_src.write_text("fake pdf")

        data_src = tmp_path / "source" / "data.csv"
        data_src.write_text("a,b\n1,2")

        output_dir = tmp_path / "organized"

        agent = CollectorAgent(config, chat_model=_mock_chat_model())
        result = agent.run(
            papers=[
                PaperEntry(
                    paper_id="test2024",
                    pdf_path=str(pdf_src),
                    data_paths=[str(data_src)],
                )
            ],
            base_dir=str(output_dir),
        )

        assert "test2024" in result
        paper_dir = result["test2024"]
        assert (paper_dir / "paper.pdf").exists()
        assert (paper_dir / "data" / "data.csv").exists()

    def test_run_warns_on_missing_pdf(self, config, tmp_path, caplog):
        import logging

        output_dir = tmp_path / "organized"

        agent = CollectorAgent(config, chat_model=_mock_chat_model())

        with caplog.at_level(logging.WARNING):
            result = agent.run(
                papers=[
                    PaperEntry(
                        paper_id="missing",
                        pdf_path="/nonexistent/paper.pdf",
                    )
                ],
                base_dir=str(output_dir),
            )

        assert "missing" in result
        assert any("PDF not found" in r.message for r in caplog.records)

    def test_run_warns_on_missing_data(self, config, tmp_path, caplog):
        import logging

        pdf_src = tmp_path / "paper.pdf"
        pdf_src.write_text("fake pdf")
        output_dir = tmp_path / "organized"

        agent = CollectorAgent(config, chat_model=_mock_chat_model())

        with caplog.at_level(logging.WARNING):
            result = agent.run(
                papers=[
                    PaperEntry(
                        paper_id="test",
                        pdf_path=str(pdf_src),
                        data_paths=["/nonexistent/data.csv"],
                    )
                ],
                base_dir=str(output_dir),
            )

        assert any("Data file not found" in r.message for r in caplog.records)

    def test_run_creates_directories(self, config, tmp_path):
        pdf_src = tmp_path / "paper.pdf"
        pdf_src.write_text("fake pdf")

        deep_dir = tmp_path / "a" / "b" / "c"

        agent = CollectorAgent(config, chat_model=_mock_chat_model())
        result = agent.run(
            papers=[
                PaperEntry(
                    paper_id="deep_test",
                    pdf_path=str(pdf_src),
                )
            ],
            base_dir=str(deep_dir),
        )

        assert (deep_dir / "deep_test" / "paper.pdf").exists()

    def test_run_copies_replication_package(self, config, tmp_path):
        pdf_src = tmp_path / "paper.pdf"
        pdf_src.write_text("fake pdf")

        rp_dir = tmp_path / "orig_package"
        rp_dir.mkdir()
        (rp_dir / "analysis.do").write_text("reg y x")

        output_dir = tmp_path / "organized"

        agent = CollectorAgent(config, chat_model=_mock_chat_model())
        result = agent.run(
            papers=[
                PaperEntry(
                    paper_id="with_rp",
                    pdf_path=str(pdf_src),
                    replication_package_path=str(rp_dir),
                )
            ],
            base_dir=str(output_dir),
        )

        rp_dest = result["with_rp"] / "replication_package"
        assert rp_dest.exists()
        assert (rp_dest / "analysis.do").exists()

    def test_run_multiple_papers(self, config, tmp_path):
        output_dir = tmp_path / "organized"

        for name in ["paper_a", "paper_b"]:
            p = tmp_path / f"{name}.pdf"
            p.write_text("fake pdf")

        agent = CollectorAgent(config, chat_model=_mock_chat_model())
        result = agent.run(
            papers=[
                PaperEntry(paper_id="a", pdf_path=str(tmp_path / "paper_a.pdf")),
                PaperEntry(paper_id="b", pdf_path=str(tmp_path / "paper_b.pdf")),
            ],
            base_dir=str(output_dir),
        )

        assert len(result) == 2
        assert (output_dir / "a" / "paper.pdf").exists()
        assert (output_dir / "b" / "paper.pdf").exists()
