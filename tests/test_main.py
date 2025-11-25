# tests/test_main.py
from typer.testing import CliRunner

from bsort.main import app

# Kita pakai library 'typer' untuk CLI (akan kita bahas di step selanjutnya)
runner = CliRunner()


def test_app_help():
    """
    Smoke Test: Cek apakah command 'bsort --help' jalan.
    Jika ini sukses, berarti entry point aplikasi tidak crash.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
