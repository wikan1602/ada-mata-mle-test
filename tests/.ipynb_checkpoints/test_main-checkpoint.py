import os

import yaml
from typer.testing import CliRunner

from ada_mata_mle.main import app, load_config

runner = CliRunner()

def test_load_config(tmp_path):
    """Test apakah fungsi pembaca config berjalan benar."""
    # 1. Buat file config bohong-bohongan (dummy)
    config_file = tmp_path / "test_settings.yaml"
    dummy_data = {
        "project_name": "test_project",
        "dataset": {"img_size": 640}
    }
    
    with open(config_file, "w") as f:
        yaml.dump(dummy_data, f)
    
    # 2. Coba baca pakai fungsi kita
    loaded_cfg = load_config(str(config_file))
    
    # 3. Cek apakah isinya sama
    assert loaded_cfg["project_name"] == "test_project"
    assert loaded_cfg["dataset"]["img_size"] == 640

def test_cli_help():
    """Test apakah command 'bsort --help' muncul tanpa error."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Ada Mata MLE Take Home Test" in result.stdout