import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_IMPORTS = [
    "import portfolio",
    "import data",
    "import alpha",
    "import backtest",
    "import execution",
    "import strategies",
]


def test_container_commands_use_quantcortex_namespace():
    for name in ("Dockerfile", "docker-compose.yml"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "import quantcortex." in text
        assert not any(legacy in text for legacy in LEGACY_IMPORTS)


def test_docker_context_excludes_local_data_and_secrets():
    entries = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert {".git", ".venv", ".env", "local_data", "reports", "submission_v*"} <= entries


def test_environment_template_keeps_redis_unset_by_default():
    result = subprocess.run(
        [
            "bash", "--noprofile", "--norc", "-c",
            'set -a; source "$1"; test "${REDIS_URL+x}" != x',
            "environment-template-test", str(REPO_ROOT / ".env.example"),
        ],
        env={},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
