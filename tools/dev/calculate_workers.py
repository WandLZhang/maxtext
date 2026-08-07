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

"""Calculates the optimal number of workers for a given test flavor."""

import argparse
import json
import math
import os
import sys


def get_max_workers_for_flavor(flavor: str) -> int:
  """Returns the maximum worker safety cap for a given test flavor."""
  if flavor.startswith("cpu-"):
    return 4
  if flavor.startswith("tpu-") or flavor.startswith("tpu7x-"):
    return 3
  if flavor.startswith("gpu-"):
    return 3
  return 1


def calculate_workers(flavor: str, baseline_data: dict[str, float] | None = None) -> tuple[int, list[int]]:
  """Calculates total workers and worker groups based on baseline data.

  Args:
    flavor: The test flavor name (e.g. 'cpu-unit', 'tpu-unit').
    baseline_data: Optional dictionary mapping baseline keys to durations (sec).

  Returns:
    A tuple of (total_workers, worker_groups).
  """
  max_workers = get_max_workers_for_flavor(flavor)

  if not baseline_data:
    return max_workers, list(range(1, max_workers + 1))

  prefix = f"{flavor}::"
  matching = [float(dur) for k, dur in baseline_data.items() if k.startswith(prefix) and isinstance(dur, (int, float))]
  test_count = len(matching)
  total_seconds = sum(matching)
  total_minutes = total_seconds / 60.0

  if test_count == 0:
    return 1, [1]

  # Never allocate more workers than available tests
  workers = min(max_workers, test_count)

  # For lightweight test suites (< 8 mins total duration), right-size worker count
  if total_minutes < 8.0:
    workers = min(workers, max(1, math.ceil(total_minutes / 4.0)))

  workers = max(1, workers)
  return workers, list(range(1, workers + 1))


def main() -> int:
  """CLI entry point for calculating worker parameters."""
  parser = argparse.ArgumentParser(description="Calculate total workers and worker groups for a test flavor.")
  parser.add_argument(
      "--flavor",
      type=str,
      required=True,
      help="Test flavor name (e.g. cpu-unit, tpu-unit)",
  )
  parser.add_argument(
      "--baseline",
      type=str,
      default=None,
      help="Path to per_test_baseline.json",
  )
  parser.add_argument(
      "--github-output",
      type=str,
      default=None,
      help="Path to GITHUB_OUTPUT file to write output variables",
  )

  args = parser.parse_args()

  baseline_data = None
  if args.baseline and os.path.isfile(args.baseline):
    try:
      with open(args.baseline, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"Warning: Failed to read baseline file '{args.baseline}': {e}",
          file=sys.stderr,
      )

  total_workers, worker_groups = calculate_workers(args.flavor, baseline_data)
  worker_groups_json = json.dumps(worker_groups)

  print(f"Flavor: {args.flavor}")
  print(f"Total Workers: {total_workers}")
  print(f"Worker Groups: {worker_groups_json}")

  if args.github_output:
    with open(args.github_output, "a", encoding="utf-8") as f:
      f.write(f"total_workers={total_workers}\n")
      f.write(f"worker_groups={worker_groups_json}\n")

  return 0


if __name__ == "__main__":
  sys.exit(main())
