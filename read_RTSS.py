#!/usr/bin/env python3

"""
Read some tags in RTSS for GET request.

You may need to install some py packages, e.g. pydicom (https://pydicom.github.io/pydicom/stable/tutorials/installation.html).
"""

import sys
import json
import argparse

import pydicom 
from collections import namedtuple

RTSSinfo = namedtuple('RTSSinfo', ('instanceUID', 'serialUID'))

def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
    
    parser.add_argument('-in', dest='input', type=str,  
                        help="Specify the RTStruct file read.")
    parser.add_argument('--out', dest='output', type=str, default="RTSS_info.json",
                        help="Specify the RTStruct file read.")
    
    return parser.parse_args()
    
def main(file, out_file):
        
    try:
        print(f'Loading the file: {file}')
        dcm = pydicom.read_file(file)
        info = RTSSinfo(instanceUID=dcm.SOPInstanceUID, 
                       serialUID=dcm.SeriesInstanceUID)

        # Output to a json file
        with open(out_file, 'w') as outfile:
            json.dump(info._asdict(), outfile)
            
        print(f'Done. Exported info to {out_file}')
        
    except Exception as e:
        print(f'Not working. Error: {e}')

if __name__ == '__main__':
    
    args = parse_arguments()
    
    main(args.input, args.output)
