"""Observable completion/heartbeat loops, safe audit checkpoints and cancellation."""
import json
import signal
import sys
import time
from concurrent.futures import wait, FIRST_COMPLETED
from threading import Event, current_thread, main_thread

from runtime import atomic_write, metrics_snapshot

STOP = Event()


def install_cancellation():
    if current_thread() is main_thread():
        def cancel(signum, frame):
            STOP.set()
            raise KeyboardInterrupt('Workflow cancellation requested')
        signal.signal(signal.SIGINT, cancel)
        signal.signal(signal.SIGTERM, cancel)


def checkpoint(path, payload):
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


class Progress:
    def __init__(self, stage, total, clock=time.monotonic):
        self.stage, self.total, self.clock = stage, total, clock
        self.started = self.last_print = clock()
        self.done = self.success = self.failures = 0
        self.initial_hits = metrics_snapshot().get(stage, {}).get('cache_hits', 0)
        self.show('start')

    def show(self, state):
        now = self.clock()
        hits = metrics_snapshot().get(self.stage, {}).get('cache_hits', 0) - self.initial_hits
        print(f'[{self.stage}] {state}: {self.done}/{self.total}, success={self.success}, '
              f'failed={self.failures}, cache_hits={hits}, elapsed={now-self.started:.1f}s', file=sys.stderr, flush=True)
        self.last_print = now

    def complete(self, success=True):
        self.done += 1
        self.success += int(success)
        self.failures += int(not success)
        if self.done % 5 == 0 or self.clock() - self.last_print >= 15 or self.done == self.total:
            self.show('progress' if self.done < self.total else 'complete')


def completed(futures, progress, chain=None, heartbeat=30):
    pending = set(futures)
    try:
        while pending:
            done, pending = wait(pending, timeout=heartbeat, return_when=FIRST_COMPLETED)
            if not done:
                progress.show('waiting')
            for future in done:
                yield future
    except BaseException:
        if chain is not None:
            chain.stopped.set()
        for future in pending:
            future.cancel()
        progress.show('stopped')
        raise
    finally:
        # Errors raised by the caller close this generator: cancel queued work as well.
        if pending:
            if chain is not None:
                chain.stopped.set()
            for future in pending:
                future.cancel()
