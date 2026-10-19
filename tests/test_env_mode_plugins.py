# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment modes contributed by installed packages through entry points."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from skillevaluator import tier3_environments
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3_environments import (
    DISABLE_ENV_MODE_PLUGINS_ENV,
    ENV_MODE_PLUGIN_GROUP,
    EnvModePlugin,
    EnvModePluginError,
)

_MODULE = "skillevaluator_test_env_plugin"
_ENV_IMPORT_PATH = f"{_MODULE}:AcmeEnvironment"
_AGENT_IMPORT_PATH = f"{_MODULE}:AcmeClaudeCode"


class _AcmeEnvironment:
    preflight_calls = 0
    preflight_error: Exception | None = None

    @classmethod
    def preflight(cls) -> None:
        cls.preflight_calls += 1
        if cls.preflight_error is not None:
            raise cls.preflight_error


def _plugin(**overrides: object) -> EnvModePlugin:
    options: dict[str, object] = {
        "import_path": _ENV_IMPORT_PATH,
        "environment_kwargs": frozenset({"endpoint", "profile", "lease_seconds", "pool"}),
        "required_environment_kwargs": ("endpoint",),
        "reserved_environment_kwargs": frozenset({"pool"}),
        "host_env_vars": frozenset({"ACME_SANDBOX_TOKEN", "ACME_SANDBOX_URL"}),
        "agent_import_paths": {"claude-code": _AGENT_IMPORT_PATH},
        "prerequisites": f"{_MODULE}:check",
        "install_hint": "Install 'skillevaluator-acme'.",
    }
    options.update(overrides)
    return EnvModePlugin(**options)  # type: ignore[arg-type]


@pytest.fixture
def plugin_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    module = types.ModuleType(_MODULE)
    module.AcmeEnvironment = _AcmeEnvironment  # type: ignore[attr-defined]
    module.AcmeClaudeCode = object  # type: ignore[attr-defined]
    module.PLUGIN = _plugin()  # type: ignore[attr-defined]
    module.check = lambda kwargs: [] if kwargs.get("endpoint") != "down" else ["acme endpoint is down"]  # type: ignore[attr-defined]
    module.make_plugin = _plugin  # type: ignore[attr-defined]
    module.broken = 42  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _MODULE, module)
    monkeypatch.delenv(DISABLE_ENV_MODE_PLUGINS_ENV, raising=False)
    _AcmeEnvironment.preflight_calls = 0
    _AcmeEnvironment.preflight_error = None
    yield module
    tier3_environments.clear_env_mode_plugin_cache()


def _install(monkeypatch: pytest.MonkeyPatch, **entries: str) -> None:
    points = [EntryPoint(name=name, value=value, group=ENV_MODE_PLUGIN_GROUP) for name, value in entries.items()]
    monkeypatch.setattr(tier3_environments, "entry_points", lambda *, group: points if group == ENV_MODE_PLUGIN_GROUP else [])
    tier3_environments.clear_env_mode_plugin_cache()


@pytest.fixture
def acme(plugin_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _install(monkeypatch, acme=f"{_MODULE}:PLUGIN")
    return plugin_module


def test_installed_plugins_extend_the_env_mode_list(acme: types.ModuleType) -> None:
    assert tier3_environments.env_mode_plugin_names() == ("acme",)
    assert tier3_environments.all_env_modes()[-1] == "acme"
    assert tier3_environments.is_env_mode("acme")
    assert not tier3_environments.is_env_mode("acme-unknown")
    assert tier3_environments.load_env_mode_plugin("docker") is None


def test_plugins_cannot_replace_built_in_modes_or_use_invalid_names(
    plugin_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, docker=f"{_MODULE}:PLUGIN", local=f"{_MODULE}:PLUGIN", Bad_Name=f"{_MODULE}:PLUGIN")

    assert tier3_environments.env_mode_plugin_names() == ()
    command = build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="d", env_mode="docker")
    assert command[command.index("--env") + 1] != _ENV_IMPORT_PATH


def test_plugins_can_be_disabled(acme: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV_MODE_PLUGINS_ENV, "1")

    assert tier3_environments.env_mode_plugin_names() == ()
    assert "Unsupported Harbor environment" in runner._check_prerequisites(env_mode="acme")[0]


def test_listing_plugins_does_not_import_them(plugin_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, acme="skillevaluator_missing_plugin_module:PLUGIN")

    assert tier3_environments.all_env_modes()[-1] == "acme"
    with pytest.raises(EnvModePluginError, match="could not be loaded"):
        tier3_environments.load_env_mode_plugin("acme")


def test_entry_point_may_name_a_factory(plugin_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, acme=f"{_MODULE}:make_plugin", junk=f"{_MODULE}:broken")

    assert tier3_environments.load_env_mode_plugin("acme") == _plugin()
    with pytest.raises(EnvModePluginError, match="must provide an EnvModePlugin"):
        tier3_environments.load_env_mode_plugin("junk")


def test_plugin_mode_dispatches_through_harbor_import_paths(acme: types.ModuleType) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="claude-code",
        job_name="acme",
        env_mode="acme",
        environment_kwargs={"endpoint": "https://sandbox.example", "lease_seconds": 600},
    )

    assert command[command.index("--env") + 1] == _ENV_IMPORT_PATH
    assert command[command.index("--agent") + 1] == _AGENT_IMPORT_PATH
    ek = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    assert ek == [f"endpoint={json.dumps('https://sandbox.example')}", "lease_seconds=600"]

    stock = build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="acme", env_mode="acme")
    assert stock[stock.index("--agent") + 1] == "opencode"


def test_plugin_agent_wrappers_are_the_only_extra_agent_paths_allowed(acme: types.ModuleType) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="claude-code",
        job_name="acme",
        env_mode="acme",
        agent_import_path=_AGENT_IMPORT_PATH,
    )
    assert command[command.index("--agent") + 1] == _AGENT_IMPORT_PATH

    with pytest.raises(ValueError, match="agent_import_path is supported only"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="claude-code",
            job_name="acme",
            env_mode="acme",
            agent_import_path="attacker.module:Agent",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown": "x"}, "does not accept environment kwarg(s): unknown"),
        ({"pool": "shared"}, "reserved for Harbor runtime policy: pool"),
        ({"network_policy": "open"}, "reserved for Harbor runtime policy: network_policy"),
        ({"session_id": "s"}, "reserved for Harbor runtime policy: session_id"),
    ],
)
def test_plugin_kwargs_follow_evaluator_policy(acme: types.ModuleType, kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="acme",
            env_mode="acme",
            environment_kwargs=kwargs,
        )


def test_plugin_kwargs_cannot_carry_secrets(acme: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="secret-bearing"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="acme",
            env_mode="acme",
            environment_kwargs={"api_token": "rt-123456"},
        )


def test_plugin_prerequisites_run_required_kwargs_hook_and_harbor_preflight(acme: types.ModuleType) -> None:
    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={}) == [
        "Harbor environment 'acme' requires non-empty string --environment-kwarg for: endpoint"
    ]
    assert _AcmeEnvironment.preflight_calls == 0

    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "down"}) == [
        "acme endpoint is down"
    ]
    assert _AcmeEnvironment.preflight_calls == 0

    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "https://up"}) == []
    assert _AcmeEnvironment.preflight_calls == 1

    _AcmeEnvironment.preflight_error = RuntimeError("ACME_SANDBOX_URL is not set")
    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "https://up"}) == [
        "Environment mode 'acme' is not ready: ACME_SANDBOX_URL is not set"
    ]


def test_plugin_load_failure_is_a_prerequisite_error_with_install_hint(
    plugin_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, acme="skillevaluator_missing_plugin_module:PLUGIN")

    [error] = runner._check_prerequisites(env_mode="acme")
    assert "could not be loaded" in error
    assert "Reinstall the package that provides environment mode 'acme'" in error


def test_plugin_host_variables_reach_harbor_and_are_protected_from_skills(acme: types.ModuleType) -> None:
    assert runner._env_mode_host_env_vars("acme") == frozenset({"ACME_SANDBOX_TOKEN", "ACME_SANDBOX_URL"})
    assert runner._env_mode_host_env_vars("daytona") == runner._HARBOR_ENV_MODE_VARS["daytona"]

    resolved, errors = runner._resolve_runtime_env({"ACME_SANDBOX_URL": "https://attacker.example"}, env_mode="acme")
    assert resolved == {}
    assert errors == ["harbor.runtime_env.ACME_SANDBOX_URL controls the host process and is not allowed"]


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "LD_PRELOAD", "SKILLEVALUATOR_X"])
def test_plugins_cannot_forward_evaluator_owned_variables(
    plugin_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    plugin_module.PLUGIN = _plugin(host_env_vars=frozenset({"ACME_SANDBOX_URL", name}))  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:PLUGIN")

    with pytest.raises(EnvModePluginError, match=f"may not forward evaluator-owned variable.*{name}"):
        build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="acme", env_mode="acme")
    assert runner._env_mode_host_env_vars("acme") == frozenset()
    assert "may not forward" in runner._check_prerequisites(env_mode="acme")[0]


def test_plugin_compose_support_gates_staged_tasks(
    acme: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = tmp_path / "task-a"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    assert runner._staged_task_environment_error("acme", [task]) is None

    (task / "environment" / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    assert "cannot run Docker Compose task 'task-a'" in str(runner._staged_task_environment_error("acme", [task]))

    acme.PLUGIN = _plugin(supports_docker_compose=True)  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:PLUGIN")
    assert runner._staged_task_environment_error("acme", [task]) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"import_path": "not-an-import-path"}, "import_path"),
        ({"prerequisites": "nope"}, "prerequisites"),
        ({"environment_kwargs": "endpoint"}, "environment_kwargs"),
        ({"required_environment_kwargs": ("region",)}, "must also be accepted: region"),
        ({"host_env_vars": frozenset({"BAD-NAME"})}, "host_env_vars"),
        ({"agent_import_paths": {"claude-code": "nope"}}, "agent_import_paths"),
    ],
)
def test_plugin_descriptors_are_validated(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(EnvModePluginError, match=message):
        _plugin(**overrides)


def test_cli_offers_installed_plugin_modes(tmp_path: Path) -> None:
    dist = tmp_path / "skillevaluator_acme_plugin-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: skillevaluator-acme-plugin\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text(f"[{ENV_MODE_PLUGIN_GROUP}]\nacme = {_MODULE}:PLUGIN\n")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from skillevaluator.cli import ENV_MODE_CHOICE; print(ENV_MODE_CHOICE.choices[-1])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "acme"
