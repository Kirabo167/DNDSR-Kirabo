#!/usr/bin/env python3
"""Execute one ACM MPI rank on its explicitly assigned physical GPU."""
import os
import sys


def main():
    devices = [int(value) for value in os.environ['DNDS_SUBOFF_RANK_DEVICES'].split(',')]
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if size != len(devices) or local_size != size or any(device < 0 for device in devices):
        raise ValueError('GPU rank mapping requires an explicit device for every rank on one host')
    if len(sys.argv) < 3:
        raise ValueError('Expected ACM executable and configuration')
    executable = sys.argv[1]
    os.execv(executable, [executable, *sys.argv[2:], '--backend', 'cuda',
                          '--gpu-device', str(devices[rank])])


if __name__ == '__main__':
    main()
