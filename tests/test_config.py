import pytest

from ace_step_adapter.config import ConfigError, load_settings

VALID = """
[server]
host = "127.0.0.1"
port = 8001

[ace_step]
project_path = "{project}"
model = "acestep-v15-turbo"
device = "auto"
quantization = "{quantization}"
offload_to_cpu = true

[tasks]
timeout_seconds = 1200

[output]
directory = "output"

[logging]
directory = "logs"
"""


def write(tmp_path, quantization="int8_weight_only", project=None):
    project = project or tmp_path
    path = tmp_path / "config.toml"
    path.write_text(VALID.format(project=str(project).replace("\\", "/"), quantization=quantization), encoding="utf-8")
    return path


def test_loads_valid_configuration(tmp_path):
    settings = load_settings(write(tmp_path))
    assert settings.port == 8001
    assert settings.quantization == "int8_weight_only"
    assert settings.offload_to_cpu is True
    assert settings.task_timeout_seconds == 1200
    assert settings.output_dir == (tmp_path / "output").resolve()


def test_quantization_none_becomes_none(tmp_path):
    assert load_settings(write(tmp_path, quantization="none")).quantization is None


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.toml")


def test_invalid_toml_is_a_config_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[server\nport = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_settings(path)


def test_unknown_quantization_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="quantization"):
        load_settings(write(tmp_path, quantization="int4"))


def test_missing_project_path_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="project_path"):
        load_settings(write(tmp_path, project=tmp_path / "does-not-exist"))


def test_port_out_of_range_is_rejected(tmp_path):
    path = write(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("port = 8001", "port = 70000"), encoding="utf-8")
    with pytest.raises(ConfigError, match="port"):
        load_settings(path)
