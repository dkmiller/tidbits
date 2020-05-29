import argparse
import logging
import os
from typeguard import typechecked
from typing import Callable
import yaml

@typechecked
def run(main: Callable[[object, logging.Logger], None]) -> None:
    '''
    Handle log levels and parsing a configuration file.
    '''
    # https://stackoverflow.com/a/5137509
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    # TODO: get callers directory.
    parser.add_argument('--config', default=os.path.join(dir_path, 'config.yml'))
    parser.add_argument('--level', default='INFO')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=args.level)
    logger = logging.getLogger()

    main(config, logger)
