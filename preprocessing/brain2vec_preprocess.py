import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

TEMPLATE = (
    Path(__file__).resolve().parent.parent / "registration" / "img_template.nii.gz"
)


def process_scans(args: Any) -> None:
    out_dir = Path(args.out_dir)
    subprocess.run(
        [
            "NiChart_DLMUSE",
            "-i",
            args.in_dir,
            "-o",
            str(out_dir),
            "-d",
            args.device,
            "--refaced-data",
        ],
        check=True,
    )

    masked_dir = out_dir / "temp_working_dir" / "s3_masked"
    aligned_dir = out_dir / "brain2vec_aligned_scans"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    for img in sorted(masked_dir.glob("*.nii.gz")):
        out = aligned_dir / f"{img.name[:-7]}_brain2vec_aligned.nii.gz"
        subprocess.run(
            ["flirt", "-in", str(img), "-ref", str(args.template), "-out", str(out)],
            check=True,
            env=env,
        )

    if not args.keep_temp:
        shutil.rmtree(out_dir / "temp_working_dir", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="process_scans",
        description="Skull-strip with NiChart_DLMUSE, then FLIRT-align to the Brain2Vec template.",
    )
    parser.add_argument(
        "--in_dir", "-i", required=True, help="Input directory with NIfTI scans"
    )
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for NiChart_DLMUSE",
    )
    parser.add_argument(
        "--template",
        default=str(TEMPLATE),
        help="Registration template (default: ../registration/img_template.nii.gz)",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep NiChart_DLMUSE's temp_working_dir",
    )
    parser.set_defaults(func=process_scans)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
