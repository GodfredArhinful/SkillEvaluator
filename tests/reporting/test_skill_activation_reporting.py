# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Never-activated trials must stay inconclusive across saved report surfaces."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from skillevaluator.evaluation.tier3_report import (
    _load_suggestions_v2,
    _validation_result_from_payload,
    build_agent_eval_payload,
)
from skillevaluator.reporting import HTMLReporter, MarkdownReporter
from skillevaluator.reporting.base import assess_tier3_evidence
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.console_ui import summarize_tier3
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.harbor.report_data import load_agent_data
from skillevaluator.tier3.result_display import render_result


def _agent(*, activated: bool = False, score: float = 0.4) -> dict:
    reward = {
        **dict.fromkeys(DEFAULT_METRICS, score),
        "entry_id": "case",
        "trial_id": "case__attempt1",
        "details": {"skill_execution": {"activation": {"passed": activated, "score": float(activated)}}},
    }
    return {
        "execution_status": "succeeded",
        "expected_attempts": 2,
        "scored_attempts": 2,
        "num_trials": 1,
        "num_trials_baseline": 1,
        "with_skill": dict.fromkeys(DEFAULT_METRICS, score),
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
        "rewards": [reward],
        "rewards_baseline": [{**reward, "trial_id": "baseline__attempt1"}],
        "lift": {"overall": {"delta": -0.4}},
        "pass_lift": {"delta": -1.0},
        "custom_with_skill": {"custom_grade": 0.2},
        "custom_without_skill": {"custom_grade": 0.9},
        "custom_lift": {"custom_grade": {"delta": -0.7}},
    }


@pytest.mark.parametrize("score", [0.2, 0.9])
def test_no_activation_suppresses_all_causal_lift_without_erasing_scores(score: float) -> None:
    payload = build_agent_eval_payload("demo", {"opencode": _agent(score=score)}, use_llm_judge=False)
    assert payload is not None
    assert payload["execution_status"] == "succeeded"
    assert payload["verdict"] == "inconclusive"
    assert payload["overall_score"] == score
    assert payload["overall_lift"] is None
    assert payload["composite_lift"] is None
    assert payload["summary"]["skill_activation"]["status"] == "never_activated"
    assert all(item["lift"] is None for item in payload["evaluators"].values())
    assert all(item["lift"] is None and item["verdict"] == "inconclusive" for item in payload["dimensions"])
    assert all(item["lift"] is None for item in payload["evaluator_cards"])
    assert payload["pass_at_k"]["lift"] == {}
    assert all("lift_scores" not in trial for trial in payload["trials"])
    assert all(item["severity"] != "fail" for item in payload["conclusions"])


def test_inconclusive_renders_consistently_in_html_markdown_cli_and_evidence_assessment() -> None:
    payload = build_agent_eval_payload("demo", {"opencode": _agent()}, use_llm_judge=False)
    result = _validation_result_from_payload(payload)
    assert result is not None
    assessment = assess_tier3_evidence([result])
    assert assessment.status == "inconclusive"
    assert not assessment.evidence_complete
    assert assessment.execution_status == "succeeded"
    assert any("INCONCLUSIVE" in text for row in summarize_tier3(result)[2] for text, _ in row.segments)
    html = HTMLReporter(include_timestamp=False).render_all([result])
    markdown = MarkdownReporter(include_timestamp=False).render_all([result])
    assert "Inconclusive: skill never activated." in html
    assert "**Verdict:** INCONCLUSIVE" in markdown
    assert "Skill never activated." in markdown
    output = io.StringIO()
    CLIReporter._print_agent_eval_tables(payload, Console(file=output, width=140, color_system=None))
    assert "Verdict: INCONCLUSIVE" in output.getvalue()


def test_mixed_agents_select_activated_evidence_even_when_unactivated_score_is_higher() -> None:
    payload = build_agent_eval_payload(
        "demo", {"unactivated": _agent(score=1.0), "activated": _agent(activated=True, score=0.8)}, use_llm_judge=False
    )
    assert payload is not None
    assert payload["best_agent"] == "activated"
    assert payload["verdict"] == "pass"
    assert payload["agents"]["unactivated"]["verdict"] == "inconclusive"
    assert payload["agents"]["unactivated"]["lift"] is None
    assert payload["overall_lift"] == 0.0


def test_mixed_agents_discard_unattributed_stale_skill_advice(tmp_path) -> None:
    agents = {"unactivated": _agent(score=1.0), "activated": _agent(activated=True, score=0.8)}
    stale_advice = [{"suggestion": "Remove the skill because it regressed quality."}]
    agent_dir = tmp_path / "unactivated"
    agent_dir.mkdir()
    (agent_dir / "findings.json").write_text(json.dumps({"suggestions_v2": stale_advice}))
    assert _load_suggestions_v2(tmp_path, agents) == []
    payload = build_agent_eval_payload("demo", agents, suggestions_v2=stale_advice, use_llm_judge=False)
    assert payload is not None
    assert payload["suggestions_v2"] == []
    assert payload["verdict"] == "pass"


def test_missing_or_incomplete_activation_evidence_does_not_assert_never_activated() -> None:
    for change in ({"rewards": []}, {"rewards_complete": False}):
        payload = build_agent_eval_payload("demo", {"opencode": {**_agent(), **change}}, use_llm_judge=False)
        assert payload is not None
        assert payload["verdict"] != "inconclusive"


def test_negative_only_activation_cases_are_not_inconclusive() -> None:
    agent = _agent()
    agent["rewards"][0]["details"]["skill_execution"] = {"should_trigger": False}
    payload = build_agent_eval_payload("demo", {"opencode": agent}, use_llm_judge=False)
    assert payload is not None
    assert payload["skill_activation"]["status"] == "not_applicable"
    assert payload["verdict"] != "inconclusive"


def test_no_baseline_does_not_turn_never_activated_scores_into_quality_verdict() -> None:
    agent = _agent(score=1.0)
    agent.pop("without_skill")
    agent.pop("rewards_baseline")
    agent["conditions"] = {"without_skill": {"execution_status": "skipped"}}
    payload = build_agent_eval_payload("demo", {"opencode": agent}, use_llm_judge=False)
    assert payload is not None
    assert payload["verdict"] == "inconclusive"
    assert payload["overall_score"] == 1.0
    assert payload["overall_lift"] is None


def test_saved_summary_preserves_census_when_trial_loading_is_incomplete(tmp_path) -> None:
    condition = tmp_path / "opencode" / "with-skill"
    condition.mkdir(parents=True)
    summary = {
        "scores": dict.fromkeys(DEFAULT_METRICS, 0.4),
        "execution_status": "succeeded",
        "num_trials": 12,
        "expected_attempts": 12,
        "scored_attempts": 12,
        "skill_activation": {
            "status": "never_activated",
            "positive_trials": 12,
            "activated_trials": 0,
            "unknown_trials": 0,
            "reason": "Skill never activated.",
        },
    }
    (condition / "summary.json").write_text(json.dumps(summary))
    agents = load_agent_data(tmp_path)
    assert agents["opencode"]["skill_activation"]["positive_trials"] == 12
    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)
    assert payload is not None and payload["verdict"] == "inconclusive"


def test_historical_saved_trials_rebuild_activation_census_for_all_report_consumers(tmp_path) -> None:
    condition = tmp_path / "opencode" / "with-skill"
    trial = condition / "trials" / "case__attempt1"
    trial.mkdir(parents=True)
    agent = _agent()
    (condition / "summary.json").write_text(
        json.dumps({"scores": agent["with_skill"], "execution_status": "succeeded", "num_trials": 1})
    )
    (trial / "reward.json").write_text(json.dumps(agent["rewards"][0]))
    agents = load_agent_data(tmp_path)
    assert agents["opencode"]["rewards_complete"]
    assert agents["opencode"]["skill_activation"]["status"] == "never_activated"


def test_terminal_does_not_restore_dimension_lift_from_raw_scores() -> None:
    agent = _agent()
    agent.update(
        skill_activation={"status": "never_activated"},
        dimensions_with_skill={"effectiveness": {"score": 0.4}},
        dimensions_without_skill={"effectiveness": {"score": 0.8}},
    )
    output = render_result({"execution_status": "succeeded", "agents": {"opencode": agent}})
    assert "INCONCLUSIVE: skill never activated" in output
    assert "OVERALL SKILL LIFT" not in output
    assert "-0.40" not in output


@pytest.mark.parametrize("activation", [None, [], "unknown", {"status": None}, {"status": []}, {"status": "unknown"}])
def test_html_tolerates_malformed_persisted_activation_summary(activation) -> None:
    payload = build_agent_eval_payload("demo", {"opencode": _agent(activated=True)}, use_llm_judge=False)
    assert payload is not None
    payload["agents"]["opencode"]["skill_activation"] = activation
    result = _validation_result_from_payload(payload)
    assert result is not None
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert "Agent Scorecards" in html
