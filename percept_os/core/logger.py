from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.theme import Theme

_THEME = Theme({
    "info": "cyan",
    "ok": "green bold",
    "warn": "yellow",
    "err": "red bold",
    "timestamp": "dim white"
})


class RunLogger:
    def __init__(self, log_file: Path):
        self.console = Console(theme=_THEME)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] [{level}] {msg}\n"
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line)

    def info(self, msg: str) -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        self.console.print(f"[timestamp][{ts}] [/timestamp][info]{msg}[/info]")
        self._write("INFO", msg)

    def ok(self, msg: str) -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        self.console.print(f"[timestamp][{ts}] [/timestamp][ok]{msg}[/ok]")
        self._write("OK", msg)

    def warn(self, msg: str) -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        self.console.print(f"[timestamp][{ts}] [/timestamp][warn]{msg}[/warn]")
        self._write("WARN", msg)

    def err(self, msg: str) -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        self.console.print(f"[timestamp][{ts}] [/timestamp][err]{msg}[/err]")
        self._write("ERR", msg)

    def blank(self) -> None:
        print()
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write("\n")