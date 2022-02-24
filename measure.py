from __future__ import annotations

import sys
from cProfile import Profile
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, ClassVar, NamedTuple


@dataclass
class Experiment:
    name: ClassVar = 'Unnamed'

    # The number of tests per experiment.
    # This MUST NOT change during a given trial (between setup calls)
    iterations: int = 1

    def setup(self):
        """
        Prepare the experiment environment.

        This is called exactly once for each experiment trial.
        """

    def prepare(self):
        """
        Prepare the experiment environment for each trial iteration.

        This is called exactly once for each experiment trial iteration
        and thus must correctly handle multiple invocations.
        """

    def run(self) -> Any:
        """
        Execute an experiment trial iteration.

        This is called exactly once for each experiment trial iteration
        and thus must correctly handle multiple invocations.
        """
        ...

    def cleanup(self):
        """
        Cleanup the experiment environment.

        This is called exactly once for each experiment trial/call to setup().
        """


class TrialResults(NamedTuple):
    prepare: float
    run: float
    iterations: int

    @property
    def total(self) -> float:
        return self.prepare + self.run  # type: ignore

    def __str__(self):
        return f'TrialResults(prepare={self.prepare}, run={self.run})'


_SENTINEL = object()


@dataclass
class ExperimentLab:
    """
    An Experiment executor.

    Each trial is profiled separately, and consists of four phases: setup,
    prepare, run, and cleanup. Only the prepare and run phases are profiled.

    The number of tests (prepare/run cycles) per trial is determined by the tests argument.

    Args:
        experiment: The experiment to benchmark.
        trials: The number of independently profiled setup/test/cleanup cycles
            to run.
        tests: The number of prepare/run cycles to run.
        profile: The profiler to use. Defaults to a cProfile instance using the
            time.perf_counter timer.
    """

    experiment: Experiment
    trials: int
    tests: int
    profile: Profile = field(default_factory=lambda: Profile(perf_counter))

    def measure_trial(self):
        experiment = self.experiment
        trials = self.trials

        experiment.setup()
        with self.profile as pr:
            for _ in range(trials):
                experiment.prepare()
                experiment.run()
        experiment.cleanup()

        # HACK: Time to use undocumented hacks
        # TODO: Replace with pstats backport
        raw_stats = pr.getstats()  # type: ignore

        prepare_src = sys.modules[self.experiment.prepare.__module__].__file__
        run_src = sys.modules[self.experiment.run.__module__].__file__
        stats = {
            entry[0].co_name: entry.totaltime
            for entry in raw_stats
            if getattr(entry[0], 'co_filename', _SENTINEL)
            in (prepare_src, run_src)
        }
        return TrialResults(stats['prepare'], stats['run'], experiment.iterations)  # type: ignore

    def measure(self):
        results = [self.measure_trial() for _ in range(self.trials)]
        return results
