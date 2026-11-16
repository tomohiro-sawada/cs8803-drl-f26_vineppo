"""
Downloading prerequisites for running experiments, including:

- Model weights: use the SFTed model weights provided by the paper
- Datasets (for training and benchmarking)

---
AGENT INSTRUCTIONS. I will have an agent implement this since this script is not intellectually interesting.

Desidarata:

- tqdm support for download, in case it takes a while
-
"""

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class DownloadConfig:
    download_root: Path = PROJECT_ROOT / ".cache"
    # if you need separate download paths
    model_download_root: Path = None
    dataset_download_root: Path = None


def main(config: DownloadConfig) -> None:
    """
    Execute download.
    """
    pass

if __name__ == "__main__":
    pass
