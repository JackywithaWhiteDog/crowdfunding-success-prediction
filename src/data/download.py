"""Download dataset

This module downloads the raw dataset from google drive

Examples:
  python download.py --help
  python download.py --url <URL> --output <output>

"""
import pathlib

import click
import gdown

@click.command()
@click.option('--url', '-u', help='URL to download dataset', required=True)
@click.option('--output', '-o', help='Output file', type=click.Path(), required=True)
def download_data(url: str, output: str) -> None:
    """Download dataset

    Args:
      url: URL to download dataset
      output: Output file

    Returns:
      None

    """
    print('Downloading from {} to {}'.format(url, output))
    pathlib.Path(output).parent \
        .mkdir(parents=True, exist_ok=True) # create directory for output file

    gdown.download(url, output, quiet=True) # download dataset

if __name__ == '__main__':
    download_data()
