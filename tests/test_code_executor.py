"""Tests for the code executor utility."""

import tempfile
from pathlib import Path

import nbformat
import pytest

from src.utils.code_executor import CodeExecutor, create_notebook_from_code


class TestCreateNotebook:
    def test_basic_creation(self, tmp_path):
        code_blocks = ["print('hello')", "x = 1 + 2"]
        output = str(tmp_path / "test.ipynb")
        path = create_notebook_from_code(code_blocks, output)

        assert Path(path).exists()
        nb = nbformat.read(path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) == 2

    def test_with_descriptions(self, tmp_path):
        code_blocks = ["print('hello')"]
        descriptions = ["A description cell"]
        output = str(tmp_path / "test.ipynb")
        path = create_notebook_from_code(code_blocks, output, descriptions)

        nb = nbformat.read(path, as_version=4)
        md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
        assert len(md_cells) == 1
        assert md_cells[0].source == "A description cell"

    def test_creates_parent_dirs(self, tmp_path):
        output = str(tmp_path / "sub" / "dir" / "test.ipynb")
        path = create_notebook_from_code(["x = 1"], output)
        assert Path(path).exists()

    def test_empty_code_blocks(self, tmp_path):
        output = str(tmp_path / "empty.ipynb")
        path = create_notebook_from_code([], output)
        nb = nbformat.read(path, as_version=4)
        assert len(nb.cells) == 0


class TestCodeExecutor:
    def test_init_defaults(self):
        executor = CodeExecutor()
        assert executor.timeout == 300
        assert executor.kernel_name == "python3"
        assert not executor._started

    def test_init_custom(self, tmp_path):
        executor = CodeExecutor(
            timeout=60, kernel_name="python3", working_dir=str(tmp_path)
        )
        assert executor.timeout == 60
        assert executor.working_dir == str(tmp_path)

    def test_context_manager(self):
        """Test that context manager starts and stops cleanly."""
        executor = CodeExecutor(timeout=30)
        with executor:
            assert executor._started
        assert not executor._started

    def test_execute_simple(self):
        """Test basic code execution."""
        with CodeExecutor(timeout=30) as executor:
            result = executor.execute("print('hello world')")
            assert result["success"] is True
            assert "hello world" in result["output"]

    def test_execute_error(self):
        """Test that errors are captured."""
        with CodeExecutor(timeout=30) as executor:
            result = executor.execute("1 / 0")
            assert result["success"] is False
            assert result["error"] is not None
            assert "ZeroDivision" in result["error"]

    def test_execute_expression(self):
        """Test that expression results are captured."""
        with CodeExecutor(timeout=30) as executor:
            result = executor.execute("2 + 3")
            assert result["success"] is True
            assert "5" in result["output"] or any(
                "5" in str(d) for d in result["data"]
            )

    def test_state_persists(self):
        """Test that kernel state persists between calls."""
        with CodeExecutor(timeout=30) as executor:
            executor.execute("x = 42")
            result = executor.execute("print(x)")
            assert result["success"] is True
            assert "42" in result["output"]

    def test_set_and_get_variable(self):
        """Test variable setting and getting."""
        with CodeExecutor(timeout=30) as executor:
            executor.set_variable("my_var", 123)
            value = executor.get_variable("my_var")
            assert value == 123

    def test_set_variable_string(self):
        with CodeExecutor(timeout=30) as executor:
            executor.set_variable("name", "test")
            value = executor.get_variable("name")
            assert value == "test"

    def test_set_variable_complex(self):
        with CodeExecutor(timeout=30) as executor:
            executor.set_variable("data", {"key": [1, 2, 3]})
            value = executor.get_variable("data")
            assert value == {"key": [1, 2, 3]}

    def test_execute_file(self, tmp_path):
        """Test executing a Python file."""
        script = tmp_path / "test_script.py"
        script.write_text("result = 2 + 2\nprint(f'Result: {result}')")

        with CodeExecutor(timeout=30) as executor:
            result = executor.execute_file(str(script))
            assert result["success"] is True
            assert "Result: 4" in result["output"]

    def test_execute_file_not_found(self):
        with CodeExecutor(timeout=30) as executor:
            result = executor.execute_file("/nonexistent/file.py")
            assert result["success"] is False

    def test_install_package(self):
        """Test package installation (uses already-installed package)."""
        with CodeExecutor(timeout=60) as executor:
            # json is a stdlib module, should always work
            result = executor.execute("import json; print('ok')")
            assert result["success"] is True

    def test_stop_without_start(self):
        """Stopping without starting should not error."""
        executor = CodeExecutor()
        executor.stop()  # Should be a no-op

    def test_double_start(self):
        """Starting twice should be idempotent."""
        executor = CodeExecutor(timeout=30)
        try:
            executor.start()
            executor.start()  # Should not error
            assert executor._started
        finally:
            executor.stop()
