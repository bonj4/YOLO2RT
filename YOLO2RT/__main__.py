import argparse
import os
import sys

from colorama import Fore

from YOLO2RT.commands import Builder,Seg_exporter,Det_exporter,TrtBuilder,InferDet,InferSeg

from YOLO2RT.version import VERSION
from YOLO2RT.info import get_latest_version

FUNCTION_MAP = {'Builder': Builder,
                'Seg_exporter': Seg_exporter,
                'Det_exporter': Det_exporter,
                'TrtBuilder': TrtBuilder,
                'InferDet':InferDet,
                'InferSeg':InferSeg}



def get_parser(subparsers, funtion_map):
    for key, value in funtion_map.items():
        cmd_parser = subparsers.add_parser(
            value.name, help=value.help, conflict_handler="resolve")
        value().fundamental_arguments(cmd_parser)
        cmd_parser.set_defaults(func=value().perform_task)

def main():
    print(f"{Fore.BLUE}Welcome to YOLO2RT\n"
          f"YOLO2RT allows you to convert your yolo models to engine.{Fore.RESET}\n")

    latest_version = get_latest_version()

    if latest_version:
        if latest_version > VERSION:
            print(f"{Fore.RED}A newer version ({latest_version}) is available!{Fore.RESET}")
            print(
                f'{Fore.RED}For latest YOLO2RT version please update with "pip install YOLO2RT --upgrade"{Fore.RESET} \n')
        else:
            print(f"{Fore.GREEN}You have the latest YOLO2RT version ({VERSION}) installed.{Fore.RESET}\n")
    else:
        print(f"{Fore.RED}Unable to fetch the latest YOLO2RT version information.{Fore.RESET}\n")


    parser = argparse.ArgumentParser(description="YOLO2RT")

    parser.add_argument(
        "--version",
        help="shows the version of YOLO2RT",
        action="version",
        version=VERSION,
    )

    subparsers = parser.add_subparsers(dest="map", help="Available commands")

    get_parser(subparsers, FUNCTION_MAP)

    args = parser.parse_args()

    if hasattr(args, 'func'):
#        try:
            args.func(vars(args))
 #       except Exception as e:
  #          parser.error(e)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
