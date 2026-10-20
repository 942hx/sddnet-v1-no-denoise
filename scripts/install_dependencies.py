"""Install only missing dependencies inside testmodel; never replace existing core packages."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path


def main():
    if Path(sys.prefix).name.lower() != "testmodel":
        raise RuntimeError(f"Refusing installation outside testmodel: {sys.executable}")
    root = Path(__file__).absolute().parents[1]

    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    def install(arguments):
        command = [sys.executable, "-m", "pip", "install", *arguments]
        with (root / "setup_commands.log").open("a", encoding="utf-8") as stream:
            stream.write(subprocess.list2cmdline(command) + "\n")
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True, cwd=root)

    torch_version, vision_version = version("torch"), version("torchvision")
    if torch_version is None and vision_version is None:
        install(["torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
    elif torch_version is None or vision_version is None:
        raise RuntimeError("Partial torch/torchvision install detected; inspect it before choosing a matching wheel.")
    else:
        print(f"Keeping existing torch={torch_version}, torchvision={vision_version}")
    constraints = root / ".core-constraints.txt"
    constraints.write_text(f"torch=={version('torch')}\ntorchvision=={version('torchvision')}\n", encoding="utf-8")
    missing = []
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            package = line.split("==", 1)[0]
            if version(package) is None:
                missing.append(line)
            else:
                print(f"Keeping installed {package}=={version(package)}")
    if missing:
        install([*missing, "--constraint", str(constraints)])
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("Missing-dependency installation finished; run the pipeline for CUDA verification.")


if __name__ == "__main__":
    main()
