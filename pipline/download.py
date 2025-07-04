from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tractseg.libs import utils


def main():
    parser = argparse.ArgumentParser(description="Download all pretrained weights (instead of them being downloaded "
                                                 "automatically on the fly when needed). Will download them to"
                                                 "`~/.tractseg/` (default). If you want them to be downloaded to a "
                                                 "different directory you can specify it by adding "
                                                 "`weights_dir=absolute_path_to_where_you_want_it` "
                                                 "to `~/.tractseg/config.txt`",
                                        epilog="Written by Jakob Wasserthal")
    args = parser.parse_args()
    # Set the weights directory to /mnt/host/BAInet
    weights_dir='/home/user/BAInet'

    utils.download_pretrained_weights("tract_segmentation", tract_definition="xtract")
    utils.download_pretrained_weights("dm_regression", tract_definition="xtract")
    utils.download_pretrained_weights("tract_segmentation")
    utils.download_pretrained_weights("endings_segmentation")
    utils.download_pretrained_weights("dm_regression")
    utils.download_pretrained_weights("peak_regression", part="Part1")
    utils.download_pretrained_weights("peak_regression", part="Part2")
    utils.download_pretrained_weights("peak_regression", part="Part3")
    utils.download_pretrained_weights("peak_regression", part="Part4")


if __name__ == '__main__':
    main()
