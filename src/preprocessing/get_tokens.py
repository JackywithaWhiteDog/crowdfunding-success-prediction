"""Generate tokens

This module generates tokens for each documents in raw dataset

Examples:
  python get_tokens.py --help
  python get_tokens.py \
    --resource <Resource file> \
    --level <Inference level, from 1 to 3> \
    --device <GPU device, -1 means not using GPU> \
    --output <output>

"""
import pathlib
import pickle

from ckip_transformers.nlp import CkipWordSegmenter
import click
import pandas as pd

@click.command()
@click.option('--resource', '-r', help='Resource file', type=click.Path(), required=True)
@click.option('--level', '-l', help='Inference level, from 1 to 3', default=1, type=int)
@click.option('--device', '-d', help='GPU device, -1 means not using GPU', default=-1, type=int)
@click.option('--output', '-o', help='Output file', type=click.Path(), required=True)
def tokenize(resource: str, level: int, device: int, output: str) -> None:
    """Tokenize project descriptions and contents

    Args:
      resource: Resource file
      level: Inference level, from 1 to 3
      device: GPU device, -1 means not using GPU
      output: Output file

    Returns:
      None

    """
    segmenter  = CkipWordSegmenter(level=level, device=device)
    df_project = pd.read_csv(resource)
    tokens = segmenter(list(map(str, df_project['content'])))
    pathlib.Path(output).parent \
        .mkdir(parents=True, exist_ok=True) # create directory for output file
    with open(output, 'wb') as file:
        pickle.dump(tokens, file)

if __name__ == '__main__':
    tokenize()
