#!/usr/bin/env python3
"""Publish a set of files and directories with rollback on partial failure."""

from __future__ import annotations

import argparse
import fcntl
import os
import secrets
import shutil
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


@contextmanager
def _publication_lock(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def publish_bundle(
    entries: Sequence[tuple[Path, Path]], *, lock_path: Path | None = None
) -> None:
    if not entries:
        raise ValueError("at least one publication entry is required")

    if any(destination.name in {"", ".."} for _, destination in entries):
        raise ValueError("publication destinations must name a file or directory")
    # Resolve parent aliases but replace a final symlink itself, not its target.
    destinations = [destination.parent.resolve() / destination.name for _, destination in entries]
    if len(set(destinations)) != len(destinations):
        raise ValueError("publication destinations must be unique")
    for index, destination in enumerate(destinations):
        if any(
            destination in other.parents or other in destination.parents
            for other in destinations[index + 1:]
        ):
            raise ValueError("publication destinations must not contain one another")

    token = f"{os.getpid()}-{secrets.token_hex(6)}"
    states = []
    for index, ((source, _), destination) in enumerate(zip(entries, destinations, strict=True)):
        source = source.absolute()
        temporary = destination.with_name(
            f".{destination.name}.quantcortex-{token}-{index}.tmp"
        )
        backup = destination.with_name(
            f".{destination.name}.quantcortex-{token}-{index}.backup"
        )
        states.append(
            {
                "source": source,
                "destination": destination,
                "temporary": temporary,
                "backup": backup,
                "installed": False,
            }
        )

    with _publication_lock(lock_path):
        try:
            for state in states:
                source = state["source"]
                destination = state["destination"]
                temporary = state["temporary"]
                backup = state["backup"]
                if not _exists(source):
                    raise FileNotFoundError(f"publication source does not exist: {source}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if _exists(temporary) or _exists(backup):
                    raise FileExistsError(
                        f"publication staging path already exists for {destination}"
                    )
                _copy(source, temporary)

            for state in states:
                destination = state["destination"]
                backup = state["backup"]
                if _exists(destination):
                    os.replace(destination, backup)

            for state in states:
                state["installed"] = True
                os.replace(state["temporary"], state["destination"])
        except BaseException as error:
            rollback_errors = []
            for state in reversed(states):
                destination = state["destination"]
                temporary = state["temporary"]
                backup = state["backup"]
                try:
                    if _exists(backup):
                        if _exists(destination):
                            _remove(destination)
                        os.replace(backup, destination)
                    elif state["installed"] and _exists(destination):
                        _remove(destination)
                    if _exists(temporary):
                        _remove(temporary)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise RuntimeError(f"artifact publication rollback failed: {details}") from error
            raise

        cleanup_errors = []
        for state in states:
            for path in (state["temporary"], state["backup"]):
                try:
                    if _exists(path):
                        _remove(path)
                except OSError as error:
                    cleanup_errors.append(str(error))
        if cleanup_errors:
            details = "; ".join(cleanup_errors)
            raise RuntimeError(
                f"artifact bundle was published but staging cleanup failed: {details}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="alternating source and destination paths",
    )
    args = parser.parse_args()
    if len(args.paths) % 2:
        parser.error("paths must be supplied as source/destination pairs")
    return args


def main() -> None:
    args = _parse_args()
    entries = list(zip(args.paths[::2], args.paths[1::2], strict=True))
    previous_handlers = {}
    handled_signals = (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM)

    def interrupt(signum: int, _frame: object) -> None:
        for handled_signal in handled_signals:
            signal.signal(handled_signal, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    for signum in handled_signals:
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        publish_bundle(entries, lock_path=args.lock)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
