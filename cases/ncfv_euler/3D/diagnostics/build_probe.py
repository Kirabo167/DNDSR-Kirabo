"""Compile an isolated diagnostic against existing libraries, without rebuilding them."""
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[4]
build = root / "build"
database = json.loads((build / "compile_commands.json").read_text())
entry = next(item for item in database if item["file"].endswith("/app/NCFV/ncfv_euler3D.cpp"))
command = shlex.split(entry["command"])
command[command.index("-o") + 1] = "/tmp/ncfv_operator_probe.o"
command[command.index("-c") + 1] = str(Path(__file__).with_name("operator_probe.cpp"))
subprocess.run(command, cwd=build, check=True)
lines = subprocess.check_output(["ninja", "-t", "commands", "ncfv_euler3D"], cwd=build, text=True).splitlines()
link = shlex.split(next(line for line in reversed(lines) if " -o app/ncfv_euler3D.exe " in line))
link = link[2:-2]  # Strip the Ninja ':' and '&&' sentinels; execute no shell.
link[link.index("-o") + 1] = "/tmp/ncfv_operator_probe"
link = ["/tmp/ncfv_operator_probe.o" if x.endswith("/app/NCFV/ncfv_euler3D.cpp.o") else x for x in link]
subprocess.run(link, cwd=build, check=True)
print("/tmp/ncfv_operator_probe")
