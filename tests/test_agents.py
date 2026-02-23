"""Tests for the agent classes.

These tests use LangChain chat model mocks to test agent logic without requiring API keys.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.base import BaseAgent
from src.agents.extractor import ExtractorAgent
from src.agents.replicator import ReplicatorAgent
from src.agents.verifier import VerifierAgent
from src.agents.explainer import ExplainerAgent
from src.models.schemas import (
    ReplicationGrade,
    ItemVerification,
    VerificationReport,
)


def _mock_chat_model(content: str) -> MagicMock:
    """Create a mock LangChain chat model that returns the given content."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=content)
    return mock


# ── BaseAgent Tests ──────────────────────────────────────────────────────


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing the abstract base class."""

    def run(self, **kwargs):
        return "ran"


class TestBaseAgent:
    def test_init(self, config):
        agent = ConcreteAgent(
            config=config, name="Test", role="tester", goal="test things"
        )
        assert agent.name == "Test"

    def test_run(self, config):
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")
        assert agent.run() == "ran"

    def test_generate(self, config):
        mock_model = _mock_chat_model("Hello from LLM")
        agent = ConcreteAgent(
            config=config, name="Test", role="tester", goal="test",
            chat_model=mock_model,
        )

        result = agent.generate("test prompt")
        assert result == "Hello from LLM"
        mock_model.invoke.assert_called_once()

    def test_generate_json(self, config):
        mock_model = _mock_chat_model('{"key": "value"}')
        agent = ConcreteAgent(
            config=config, name="Test", role="tester", goal="test",
            chat_model=mock_model,
        )

        result = agent.generate_json("give me json")
        assert result == {"key": "value"}

    def test_generate_json_extracts_from_text(self, config):
        mock_model = _mock_chat_model('Here is the JSON:\n{"key": "value"}\nDone.')
        agent = ConcreteAgent(
            config=config, name="Test", role="tester", goal="test",
            chat_model=mock_model,
        )

        result = agent.generate_json("give me json")
        assert result == {"key": "value"}

    def test_generate_json_invalid(self, config):
        mock_model = _mock_chat_model("not json at all")
        agent = ConcreteAgent(
            config=config, name="Test", role="tester", goal="test",
            chat_model=mock_model,
        )

        with pytest.raises(ValueError, match="No JSON found"):
            agent.generate_json("give me json")


# ── ExtractorAgent Tests ─────────────────────────────────────────────────


class TestExtractorAgent:
    def test_init(self, config):
        mock_model = _mock_chat_model("")
        agent = ExtractorAgent(config, chat_model=mock_model)
        assert agent.name == "Extractor"

    @patch("src.agents.extractor.extract_text_from_pdf")
    @patch("src.agents.extractor.extract_table_captions")
    @patch("src.agents.extractor.extract_figure_captions")
    def test_run(self, mock_fig_caps, mock_tbl_caps, mock_extract, config):
        mock_extract.return_value = "Methods\nWe use OLS regression..."
        mock_tbl_caps.return_value = [
            {"table_number": "Table 1", "caption": "Summary Stats"}
        ]
        mock_fig_caps.return_value = []

        llm_response = {
            "paper_id": "test",
            "title": "Test Paper",
            "research_questions": ["Does X affect Y?"],
            "data_description": "Panel data",
            "data_context": "Natural experiment",
            "data_processing_steps": [],
            "tables": [
                {
                    "table_number": "Table 1",
                    "caption": "Summary Stats",
                    "column_names": ["Mean"],
                    "row_names": ["X"],
                    "regression_specs": [],
                }
            ],
            "figures": [],
        }

        mock_model = _mock_chat_model(json.dumps(llm_response))
        agent = ExtractorAgent(config, chat_model=mock_model)

        result = agent.run(paper_path="test.pdf", paper_id="test")
        assert result.paper_id == "test"
        assert len(result.tables) == 1

    def test_validate_no_results_warns_on_pvalues(self, config):
        mock_model = _mock_chat_model("")
        agent = ExtractorAgent(config, chat_model=mock_model)
        from src.models.schemas import PaperSummary

        # This should log a warning but not raise
        summary = PaperSummary(
            paper_id="test",
            data_description="Data with p < 0.05 values",
            data_context="Context",
        )
        agent._validate_no_results(summary)  # Should not raise


# ── VerifierAgent Tests ──────────────────────────────────────────────────


class TestVerifierAgent:
    def test_init(self, config):
        mock_model = _mock_chat_model("")
        agent = VerifierAgent(config, chat_model=mock_model)
        assert agent.name == "Verifier"
        assert agent.tolerance == 0.01

    def test_calculate_overall_grade_all_a(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        grades = [ReplicationGrade.A, ReplicationGrade.A, ReplicationGrade.A]
        assert agent._calculate_overall_grade(grades) == ReplicationGrade.A

    def test_calculate_overall_grade_mixed(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        grades = [ReplicationGrade.A, ReplicationGrade.B, ReplicationGrade.C]
        result = agent._calculate_overall_grade(grades)
        assert result == ReplicationGrade.B  # avg = 3.0

    def test_calculate_overall_grade_low(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        grades = [ReplicationGrade.D, ReplicationGrade.F, ReplicationGrade.F]
        result = agent._calculate_overall_grade(grades)
        assert result in (ReplicationGrade.D, ReplicationGrade.F)

    def test_calculate_overall_grade_empty(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        assert agent._calculate_overall_grade([]) == ReplicationGrade.F

    def test_generate_summary(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        verifications = [
            ItemVerification(
                item_id="Table 1", item_type="table",
                grade=ReplicationGrade.A, comparison_notes="Perfect match",
            ),
            ItemVerification(
                item_id="Table 2", item_type="table",
                grade=ReplicationGrade.D, comparison_notes="Major differences found",
            ),
        ]
        summary = agent._generate_summary(verifications, ReplicationGrade.B)
        assert "Overall replication grade: B" in summary
        assert "Total items verified: 2" in summary
        assert "Table 2" in summary

    def test_extract_table_section(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        text = "Some text\nTable 1 shows the results\nMore detail about Table 1\nOther stuff"
        section = agent._extract_table_section(text, "Table 1")
        assert "Table 1" in section

    def test_extract_table_section_not_found(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        section = agent._extract_table_section("no tables here", "Table 99")
        assert "not found" in section.lower() or section == ""

    def test_verify_table_failed_execution(self, config, replication_results):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        from src.models.schemas import GeneratedTable

        failed_table = GeneratedTable(
            table_number="Table 1",
            data={},
            code_reference="Table 1",
            execution_success=False,
            error_message="ImportError",
        )
        result = agent._verify_table(failed_table, "paper text", [])
        assert result.grade == ReplicationGrade.F

    def test_verify_figure_failed_execution(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        from src.models.schemas import GeneratedFigure

        failed_fig = GeneratedFigure(
            figure_number="Figure 1",
            file_path="/tmp/nonexistent.png",
            code_reference="Figure 1",
            execution_success=False,
            error_message="Error",
        )
        result = agent._verify_figure(failed_fig, "paper text", "paper.pdf")
        assert result.grade == ReplicationGrade.F

    def test_verify_figure_file_not_found(self, config):
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        from src.models.schemas import GeneratedFigure

        fig = GeneratedFigure(
            figure_number="Figure 1",
            file_path="/tmp/definitely_not_here.png",
            code_reference="Figure 1",
            execution_success=True,
        )
        result = agent._verify_figure(fig, "paper text", "paper.pdf")
        assert result.grade == ReplicationGrade.F
        assert "not found" in result.comparison_notes.lower()

    def test_vision_uses_raw_client(self, config):
        """Vision methods should use raw API clients, not LangChain."""
        agent = VerifierAgent(config, chat_model=_mock_chat_model(""))
        assert agent._vision_provider == "openai"
        assert agent._vision_client is None  # Lazy initialized


# ── ExplainerAgent Tests ─────────────────────────────────────────────────


class TestExplainerAgent:
    def test_init(self, config):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        assert agent.name == "Explainer"

    def test_load_replication_package_none(self, config):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        assert agent._load_replication_package(None) is None

    def test_load_replication_package_missing(self, config):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        assert agent._load_replication_package("/nonexistent/path") is None

    def test_load_replication_package(self, config, tmp_path):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        # Create some files
        (tmp_path / "analysis.py").write_text("import pandas as pd\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "helper.R").write_text("library(tidyverse)\n")

        package = agent._load_replication_package(str(tmp_path))
        assert package is not None
        assert len(package["files"]) == 2

    def test_get_method_summary_found(self, config, paper_summary):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        result = agent._get_method_summary(paper_summary, "Table 1")
        assert "Summary Statistics" in result

    def test_get_method_summary_not_found(self, config, paper_summary):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        result = agent._get_method_summary(paper_summary, "Table 99")
        assert "not found" in result.lower()

    def test_get_replication_code_found(self, config, replication_results):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        code = agent._get_replication_code(replication_results, "Table 1")
        assert "describe" in code

    def test_get_replication_code_not_found(self, config, replication_results):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        code = agent._get_replication_code(replication_results, "Table 99")
        assert "not found" in code.lower()

    def test_generate_overall_assessment_no_analyses(self, config):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        report = VerificationReport(
            paper_id="test",
            overall_grade=ReplicationGrade.A,
            item_verifications=[],
            summary="All good",
        )
        result = agent._generate_overall_assessment([], report)
        assert "fully successful" in result.lower()

    def test_generate_recommendations_software(self, config):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        from src.models.schemas import DiscrepancyAnalysis

        analyses = [
            DiscrepancyAnalysis(
                item_id="Table 1",
                grade=ReplicationGrade.C,
                description_of_discrepancy="Different results",
                likely_causes=["Different software implementation"],
                is_identifiable=True,
                fault_attribution="replicator",
                confidence="medium",
            )
        ]
        recs = agent._generate_recommendations(analyses)
        assert any("software" in r.lower() for r in recs)

    def test_generate_recommendations_default(self, config):
        agent = ExplainerAgent(config, chat_model=_mock_chat_model(""))
        from src.models.schemas import DiscrepancyAnalysis

        analyses = [
            DiscrepancyAnalysis(
                item_id="Table 1",
                grade=ReplicationGrade.D,
                description_of_discrepancy="Unknown",
                likely_causes=["Unknown cause"],
                is_identifiable=False,
                fault_attribution="unclear",
                confidence="low",
            )
        ]
        recs = agent._generate_recommendations(analyses)
        assert len(recs) > 0


# ── ReplicatorAgent Tests ────────────────────────────────────────────────


class TestReplicatorAgent:
    def test_init(self, config):
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        assert agent.name == "Replicator"

    def test_extract_code_python_block(self, config):
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        response = "Here is the code:\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```\nDone."
        code, lang = agent._extract_code(response)
        assert "import pandas" in code
        assert "```" not in code
        assert lang == "python"

    def test_extract_code_r_block(self, config):
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        response = "Here is R code:\n```r\nlibrary(tidyverse)\ndf <- read_csv('data.csv')\n```"
        code, lang = agent._extract_code(response)
        assert "library(tidyverse)" in code
        assert lang == "r"

    def test_extract_code_generic_block(self, config):
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        response = "```\nprint('hello')\n```"
        code, lang = agent._extract_code(response)
        assert code == "print('hello')"
        assert lang == "python"

    def test_extract_code_no_block(self, config):
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        response = "x = 1 + 2"
        code, lang = agent._extract_code(response)
        assert code == "x = 1 + 2"
        assert lang == "python"

    def test_generate_setup_code(self, config, paper_summary):
        mock_response = '```python\nimport pandas as pd\ndf = pd.read_csv("data/test.csv")\nprint(df.shape)\n```'
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(mock_response))
        code = agent._generate_setup_code("data/test.csv", paper_summary)
        assert code.language == "python"
        assert "pandas" in code.code
        assert code.execution_order == 0

    def test_execute_r_code_no_rpy2(self, config):
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        # This will fail if rpy2 is not installed, which is expected
        result = agent._execute_r_code("print('hello')")
        # Either rpy2 is installed and it works, or we get the expected error
        if not result["success"]:
            assert "rpy2" in result["error"].lower() or "error" in result["error"].lower()

    def test_capture_data_schema(self, config):
        """_capture_data_schema stores dtypes, shape, and head output."""
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        mock_executor = MagicMock()
        mock_executor.execute.side_effect = [
            {"success": True, "output": "(100, 5)\n"},
            {"success": True, "output": "col_a    int64\ncol_b    float64\n"},
            {"success": True, "output": "   col_a  col_b\n0      1    2.0\n"},
        ]
        agent.executor = mock_executor
        agent._capture_data_schema()

        assert "(100, 5)" in agent._data_schema_info
        assert "col_a" in agent._data_schema_info
        assert "col_b" in agent._data_schema_info
        assert mock_executor.execute.call_count == 3

    def test_capture_data_schema_truncates_wide(self, config):
        """Wide DataFrames (>80 columns) get truncated."""
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        mock_executor = MagicMock()
        # Generate 100 column lines
        many_cols = "\n".join(f"col_{i}    int64" for i in range(100))
        mock_executor.execute.side_effect = [
            {"success": True, "output": "(100, 100)\n"},
            {"success": True, "output": many_cols + "\n"},
            {"success": True, "output": "sample\n"},
        ]
        agent.executor = mock_executor
        agent._capture_data_schema()

        assert "20 more columns" in agent._data_schema_info

    def test_strip_ansi(self, config):
        """_strip_ansi removes ANSI escape codes."""
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        colored = "\x1b[31mKeyError\x1b[0m: 'missing_col'"
        assert agent._strip_ansi(colored) == "KeyError: 'missing_col'"

    def test_retry_on_error(self, config, paper_summary, tmp_path):
        """Failed code triggers retry with error feedback via CODE_FIX_PROMPT."""
        # First LLM call: original code generation → bad code
        # Second LLM call: fix prompt → good code
        responses = [
            '```python\nprint(df["bad_col"])\n```',
            '```python\nprint(df.describe())\n```',
        ]
        call_count = {"n": 0}

        def mock_invoke(messages):
            idx = call_count["n"]
            call_count["n"] += 1
            return AIMessage(content=responses[idx])

        mock_model = MagicMock()
        mock_model.invoke.side_effect = mock_invoke

        agent = ReplicatorAgent(config, chat_model=mock_model)
        agent._data_schema_info = "col_a    int64\ncol_b    float64"

        mock_executor = MagicMock()
        # First execute fails, second succeeds, third is CSV save
        mock_executor.execute.side_effect = [
            {"success": False, "output": "", "error": "KeyError: 'bad_col'"},
            {"success": True, "output": "summary stats", "error": None},
            {"success": True, "output": "", "error": None},  # CSV save
        ]
        agent.executor = mock_executor

        exec_logger = MagicMock()
        result = agent._replicate_table(
            paper_summary.tables[0], paper_summary,
            "data/test.csv", tmp_path, exec_logger,
        )

        assert result is not None
        assert result["table"].execution_success is True
        # LLM called twice: original + fix
        assert mock_model.invoke.call_count == 2
        # Executor called at least twice: original + fixed code (+ CSV save)
        assert mock_executor.execute.call_count >= 2

    def test_retries_capped(self, config, paper_summary, tmp_path):
        """Retries stop after max_retries even if code keeps failing."""
        mock_model = _mock_chat_model('```python\nprint(df["bad"])\n```')
        agent = ReplicatorAgent(config, chat_model=mock_model)
        agent._data_schema_info = "col_a    int64"

        mock_executor = MagicMock()
        # All executions fail
        mock_executor.execute.return_value = {
            "success": False, "output": "", "error": "KeyError: 'bad'",
        }
        agent.executor = mock_executor

        exec_logger = MagicMock()
        result = agent._replicate_table(
            paper_summary.tables[0], paper_summary,
            "data/test.csv", tmp_path, exec_logger,
        )

        assert result is not None
        assert result["table"].execution_success is False
        # 1 original + max_retries fix attempts + (possibly R attempt)
        max_retries = config.execution.max_retries
        # LLM: 1 original + max_retries fixes + 1 R attempt = max_retries + 2
        assert mock_model.invoke.call_count == max_retries + 2

    def test_data_schema_in_prompt(self, config, paper_summary, tmp_path):
        """The data schema is injected into the code generation prompt."""
        captured_prompts = []

        def capture_invoke(messages):
            # Capture the human message content
            for msg in messages:
                if hasattr(msg, "content") and "data_schema" not in str(type(msg)):
                    captured_prompts.append(msg.content)
            return AIMessage(content='```python\nprint(df.describe())\n```')

        mock_model = MagicMock()
        mock_model.invoke.side_effect = capture_invoke

        agent = ReplicatorAgent(config, chat_model=mock_model)
        agent._data_schema_info = "col_x    float64\ncol_y    int64"

        mock_executor = MagicMock()
        mock_executor.execute.return_value = {
            "success": True, "output": "ok", "error": None,
        }
        agent.executor = mock_executor

        exec_logger = MagicMock()
        agent._replicate_table(
            paper_summary.tables[0], paper_summary,
            "data/test.csv", tmp_path, exec_logger,
        )

        # The prompt sent to the LLM should contain the schema info
        assert any("col_x" in p for p in captured_prompts)
        assert any("col_y" in p for p in captured_prompts)

    def test_figure_absolute_path(self, config, paper_summary, tmp_path):
        """Figure save paths are absolute in the generated spec."""
        captured_prompts = []

        def capture_invoke(messages):
            for msg in messages:
                if hasattr(msg, "content"):
                    captured_prompts.append(msg.content)
            return AIMessage(content='```python\nimport matplotlib.pyplot as plt\nplt.plot([1,2])\nplt.savefig("/abs/path.png")\n```')

        mock_model = MagicMock()
        mock_model.invoke.side_effect = capture_invoke

        agent = ReplicatorAgent(config, chat_model=mock_model)
        agent._data_schema_info = "col_a    int64"

        mock_executor = MagicMock()
        mock_executor.execute.return_value = {
            "success": True, "output": "", "error": None,
        }
        agent.executor = mock_executor

        exec_logger = MagicMock()
        result = agent._replicate_figure(
            paper_summary.figures[0], paper_summary,
            "data/test.csv", tmp_path, exec_logger,
        )

        # The figure path in the result should be absolute
        assert result is not None
        fig_path = result["figure"].file_path
        assert Path(fig_path).is_absolute()
        # The prompt should mention "EXACT absolute path"
        assert any("EXACT absolute path" in p for p in captured_prompts)

    def test_script_saved(self, config, paper_summary, tmp_path):
        """_save_script writes a standalone .py file with header and setup."""
        agent = ReplicatorAgent(config, chat_model=_mock_chat_model(""))
        agent._setup_code_str = "import pandas as pd\ndf = pd.read_csv('data.csv')"

        agent._save_script(
            code="print(df.describe())",
            label="table_1",
            output_path=tmp_path,
            paper_id="test_paper",
            success=True,
        )

        script_path = tmp_path / "table_1.py"
        assert script_path.exists()
        content = script_path.read_text()
        assert "# Paper: test_paper" in content
        assert "# Status: SUCCESS" in content
        assert "import pandas as pd" in content
        assert "print(df.describe())" in content
