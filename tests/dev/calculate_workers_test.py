# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for tools/dev/calculate_workers.py."""

import json
import subprocess
import sys
from tools.dev.calculate_workers import calculate_workers
from tools.dev.calculate_workers import get_max_workers_for_flavor


def test_get_max_workers_for_flavor():
  """Tests device-specific maximum worker safety caps."""
  assert get_max_workers_for_flavor("cpu-unit") == 4
  assert get_max_workers_for_flavor("cpu-integration") == 4
  assert get_max_workers_for_flavor("cpu-post-training-unit") == 4
  assert get_max_workers_for_flavor("tpu-unit") == 3
  assert get_max_workers_for_flavor("tpu-integration") == 3
  assert get_max_workers_for_flavor("tpu7x-unit") == 3
  assert get_max_workers_for_flavor("gpu-unit") == 3
  assert get_max_workers_for_flavor("gpu-integration") == 3
  assert get_max_workers_for_flavor("unknown-flavor") == 1


def test_calculate_workers_no_baseline():
  """Tests fallback behavior when baseline data is not provided."""
  workers, groups = calculate_workers("cpu-unit", None)
  assert workers == 4
  assert groups == [1, 2, 3, 4]

  workers, groups = calculate_workers("tpu-unit", None)
  assert workers == 3
  assert groups == [1, 2, 3]

  workers, groups = calculate_workers("unknown-flavor", None)
  assert workers == 1
  assert groups == [1]


def test_calculate_workers_zero_matching_tests():
  """Tests behavior when baseline exists but has 0 matching tests for flavor."""
  baseline = {
      "tpu-unit::tests.unit.test_a": 10.0,
  }
  workers, groups = calculate_workers("cpu-unit", baseline)
  assert workers == 1
  assert groups == [1]


def test_calculate_workers_heavy_suite():
  """Tests worker allocation for heavy suites (> 8 mins total duration)."""
  baseline = {f"cpu-unit::test_{i}": 60.0 for i in range(20)}  # 20 mins total
  workers, groups = calculate_workers("cpu-unit", baseline)
  assert workers == 4
  assert groups == [1, 2, 3, 4]

  baseline_tpu = {f"tpu-unit::test_{i}": 60.0 for i in range(15)}  # 15 mins total
  workers, groups = calculate_workers("tpu-unit", baseline_tpu)
  assert workers == 3
  assert groups == [1, 2, 3]


def test_calculate_workers_lightweight_suite():
  """Tests worker right-sizing for lightweight test suites (< 8 mins total)."""
  # 3 tests taking 40s each = 120s (2.0 mins total)
  baseline = {
      "cpu-post-training-integration::test_1": 40.0,
      "cpu-post-training-integration::test_2": 40.0,
      "cpu-post-training-integration::test_3": 40.0,
  }
  workers, groups = calculate_workers("cpu-post-training-integration", baseline)
  assert workers == 1
  assert groups == [1]

  # 6 tests taking 50s each = 300s (5.0 mins total) -> ceil(5.0 / 4.0) = 2 workers
  baseline_medium = {f"cpu-post-training-unit::test_{i}": 50.0 for i in range(6)}
  workers, groups = calculate_workers("cpu-post-training-unit", baseline_medium)
  assert workers == 2
  assert groups == [1, 2]


def test_calculate_workers_fewer_tests_than_max_workers():
  """Tests that worker count never exceeds the number of tests."""
  # 2 tests taking 500s each = 1000s (16.6 mins total) -> max_workers would be 4, but only 2 tests!
  baseline = {
      "cpu-unit::test_heavy_1": 500.0,
      "cpu-unit::test_heavy_2": 500.0,
  }
  workers, groups = calculate_workers("cpu-unit", baseline)
  assert workers == 2
  assert groups == [1, 2]


def test_main_cli(tmp_path):
  """Tests the CLI entry point with --github-output."""
  baseline_file = tmp_path / "baseline.json"
  baseline_data = {
      "tpu-unit::test_1": 600.0,
      "tpu-unit::test_2": 600.0,
  }
  baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

  github_output = tmp_path / "github_output.txt"

  cmd = [
      sys.executable,
      "-m",
      "tools.dev.calculate_workers",
      "--flavor",
      "tpu-unit",
      "--baseline",
      str(baseline_file),
      "--github-output",
      str(github_output),
  ]

  res = subprocess.run(cmd, capture_output=True, text=True, check=True)
  assert res.returncode == 0
  assert "Total Workers: 2" in res.stdout
  assert "Worker Groups: [1, 2]" in res.stdout

  content = github_output.read_text(encoding="utf-8")
  assert "total_workers=2\n" in content
  assert "worker_groups=[1, 2]\n" in content
