# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html


def get_prefix(img_name: str) -> str:
    """
    Returns the prefix(the name or label) of the images. Note that, the images
    have a suffix of .nii.gz and the dlicv and LPS orientation labels.
    """
    return str(img_name[: len(img_name) - 28])
