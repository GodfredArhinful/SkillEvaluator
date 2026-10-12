# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed tool results cannot prove a successful SKILL.md read."""

from __future__ import annotations

import json
from types import ModuleType

import pytest

from skillevaluator.tier3.eval_core import atif_helpers, checks
from skillevaluator.tier3.eval_core.log_converters import synthetic_trajectory_from_claude_stream_jsonl


@pytest.fixture(params=("host", "verifier"))
def grader(request: pytest.FixtureRequest, harbor_eval_template: ModuleType) -> tuple[ModuleType, ModuleType]:
    if request.param == "host":
        return atif_helpers, checks
    return harbor_eval_template, harbor_eval_template


def _claude_read(*, is_error: bool, content: str) -> dict:
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "/skills/demo-skill/SKILL.md"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-1",
                        "is_error": is_error,
                        "content": content,
                    }
                ]
            },
        },
    ]
    trajectory = synthetic_trajectory_from_claude_stream_jsonl("\n".join(json.dumps(event) for event in events))
    assert trajectory is not None
    return trajectory


def test_claude_failed_read_cannot_receive_perfect_execution_credit(grader: tuple[ModuleType, ModuleType]) -> None:
    extractor, scorer = grader
    trajectory = _claude_read(is_error=True, content="Error: File does not exist: /skills/demo-skill/SKILL.md")
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert calls[0]["is_error"] is True
    execution = scorer.score_skill_execution(calls, "demo-skill")
    assert execution["details"]["activation"]["score"] == 0.0
    assert execution["details"]["workflow_order"]["score"] == 0.0
    assert execution["score"] < 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 0.0


def test_successful_skill_content_can_discuss_errors(grader: tuple[ModuleType, ModuleType]) -> None:
    extractor, scorer = grader
    calls = extractor.extract_tool_calls_as_dicts(
        _claude_read(is_error=False, content="# Demo skill\nError: File does not exist. Handle this by retrying.")
    )

    assert scorer.score_skill_execution(calls, "demo-skill")["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 1.0


@pytest.mark.parametrize("action", ("Read", "Bash"))
def test_failed_read_does_not_satisfy_workflow_before_execution(
    grader: tuple[ModuleType, ModuleType],
    action: str,
) -> None:
    _, scorer = grader
    read = {
        "action": action,
        "action_input": {"file_path": "/skills/demo-skill/SKILL.md"}
        if action == "Read"
        else {"command": "cat /skills/demo-skill/SKILL.md"},
        "observation": "Unable to load /skills/demo-skill/SKILL.md",
        "is_error": True,
    }
    execute = {"action": "Bash", "action_input": {"command": "python scripts/run.py"}, "observation": "ok"}

    assert scorer.check_activation([read], "demo-skill")["score"] == 0.0
    assert scorer.check_routing([read], "demo-skill")["score"] == 0.0
    assert scorer.check_workflow_order([read, execute], expected_skill="demo-skill")["score"] == 0.0
    retry = {**read, "observation": "# Demo skill", "is_error": False}
    assert scorer.check_workflow_order([read, retry, execute], expected_skill="demo-skill")["score"] == 1.0
    assert scorer.check_routing([read, retry], "demo-skill")["score"] == 1.0


def test_native_atif_error_is_matched_to_its_own_call(grader: tuple[ModuleType, ModuleType]) -> None:
    from harbor.models.trajectories.trajectory import Trajectory

    extractor, scorer = grader
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "claude-code", "version": "test"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "",
                    "tool_calls": [
                        {
                            "tool_call_id": "wrong",
                            "function_name": "Read",
                            "arguments": {"file_path": "/skills/wrong/SKILL.md"},
                        },
                        {
                            "tool_call_id": "right",
                            "function_name": "Read",
                            "arguments": {"file_path": "/skills/demo-skill/SKILL.md"},
                        },
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "right",
                                "content": "# Demo skill",
                                "extra": {"tool_result_is_error": False},
                            },
                            {
                                "source_call_id": "wrong",
                                "content": "Missing file",
                                "extra": {"tool_result_is_error": True},
                            },
                        ]
                    },
                }
            ],
        }
    ).to_json_dict()
    calls = extractor.extract_tool_calls_as_dicts(trajectory)

    assert calls[0]["is_error"] is True
    assert not calls[1].get("is_error")
    assert scorer.score_skill_execution(calls, "demo-skill")["score"] == 1.0
    assert scorer.check_routing(calls, "demo-skill")["score"] == 1.0
