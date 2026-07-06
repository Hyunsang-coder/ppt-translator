"""Unit tests for the team translation-rules loader (WP-C1)."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from src.utils.rules_loader import RulesLoader

_FIXTURE = Path(__file__).parent / "fixtures" / "translation-rules-sample.json"


def _load_fixture() -> dict:
    return RulesLoader().load_rules(io.BytesIO(_FIXTURE.read_bytes()))


class LoadRulesTestCase(unittest.TestCase):
    def test_load_valid_rules(self) -> None:
        rules = _load_fixture()
        self.assertIn("kr_target_rules", rules)
        self.assertIn("en_target_rules", rules)
        self.assertIn("bidirectional", rules)

    def test_malformed_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            RulesLoader().load_rules(io.BytesIO(b"{not json"))

    def test_missing_buckets_raises(self) -> None:
        payload = json.dumps({"meta": {}, "something_else": []}).encode("utf-8")
        with self.assertRaises(ValueError):
            RulesLoader().load_rules(io.BytesIO(payload))

    def test_oversized_file_raises(self) -> None:
        big = io.BytesIO(b"x" * (RulesLoader.MAX_FILE_SIZE_MB * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            RulesLoader().load_rules(big)


class SliceForDirectionTestCase(unittest.TestCase):
    def test_kr_target_includes_kr_and_bidirectional(self) -> None:
        rules = _load_fixture()
        entries = RulesLoader.slice_for_direction(rules, "한국어")
        ids = {e["id"] for e in entries}
        self.assertIn("test-gun-sound", ids)
        self.assertIn("test-bidi", ids)
        self.assertNotIn("test-beauty-corner", ids)  # en bucket excluded

    def test_en_target_includes_en_and_bidirectional(self) -> None:
        rules = _load_fixture()
        entries = RulesLoader.slice_for_direction(rules, "영어")
        ids = {e["id"] for e in entries}
        self.assertIn("test-beauty-corner", ids)
        self.assertIn("test-bidi", ids)
        self.assertNotIn("test-gun-sound", ids)  # kr bucket excluded

    def test_unknown_direction_uses_bidirectional_only(self) -> None:
        rules = _load_fixture()
        entries = RulesLoader.slice_for_direction(rules, "독일어")
        ids = {e["id"] for e in entries}
        self.assertEqual(ids, {"test-bidi"})

    def test_none_rules_returns_empty(self) -> None:
        self.assertEqual(RulesLoader.slice_for_direction(None, "한국어"), [])


class FormatTeamRulesTestCase(unittest.TestCase):
    def test_none_when_empty(self) -> None:
        self.assertEqual(RulesLoader.format_team_rules(None, "한국어"), "None")

    def test_red_rule_includes_example(self) -> None:
        rules = _load_fixture()
        out = RulesLoader.format_team_rules(rules, "한국어")
        self.assertIn("총기 사운드", out)
        self.assertIn("avoid:", out)
        # Red rule example pair is rendered.
        self.assertIn("사격음 볼륨", out)
        self.assertIn("총기 사운드 볼륨", out)

    def test_yellow_rule_omits_example(self) -> None:
        rules = _load_fixture()
        out = RulesLoader.format_team_rules(rules, "한국어")
        # The yellow rule's summary appears, but its example pair does not.
        self.assertIn("약한 제안 규칙", out)
        self.assertNotIn("나쁜표현 예시", out)

    def test_locked_term_marker_rendered(self) -> None:
        rules = _load_fixture()
        out = RulesLoader.format_team_rules(rules, "영어")
        self.assertIn("[LOCKED TERM, use exactly: Beauty Corner]", out)


class LockedTermsTestCase(unittest.TestCase):
    def test_extracts_red_locked_terms(self) -> None:
        rules = _load_fixture()
        locked = RulesLoader.locked_terms(rules, "영어")
        self.assertEqual(locked, {"test-beauty-corner": "Beauty Corner"})

    def test_no_locked_terms_for_kr_direction(self) -> None:
        rules = _load_fixture()
        self.assertEqual(RulesLoader.locked_terms(rules, "한국어"), {})


if __name__ == "__main__":
    unittest.main()
