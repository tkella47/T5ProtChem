import sys

import torch
from pytorch_lightning.cli import LightningCLI

import datamodule  # noqa: F401
import model  # noqa: F401


# We are going to refactor this to support compile
def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    print(sys.argv)
    cli_main()
