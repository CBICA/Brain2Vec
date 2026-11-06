import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TEMPLATE = (
    Path(__file__).resolve().parent.parent / "registration" / "img_template.nii.gz"
)


def align_scan(img: Path, aligned_dir: Path, template: str, env: dict) -> Path:
    out = aligned_dir / f"{img.name[:-7]}_brain2vec_aligned.nii.gz"
    subprocess.run(
        ["flirt", "-in", str(img), "-ref", str(template), "-out", str(out)],
        check=True,
        env=env,
    )
    return out


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
    imgs = sorted(masked_dir.glob("*.nii.gz"))
    workers = max(1, min(args.workers, len(imgs))) if imgs else 1

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(align_scan, img, aligned_dir, args.template, env): img
            for img in imgs
        }
        for _, fut in enumerate(as_completed(futures), 1):
            img = futures[fut]
            try:
                fut.result()
            except subprocess.CalledProcessError:
                failed.append(img.name)

    if failed:
        print(f"{len(failed)} scan(s) failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

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
        "--workers",
        "-w",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel FLIRT jobs (default: all CPU cores)",
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
