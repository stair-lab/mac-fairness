#!/usr/bin/env python3
"""Compress experiment results and bookkeeping directories.

Recursively compresses subdirectories under bookkeeping/ and experiment/
into standalone .tar.zst files.

Usage:
    python script/compress_results.py                    # Compress all
    python script/compress_results.py --dry-run          # Show what would be compressed
    python script/compress_results.py --depth 2          # Compress at depth 2 (default)
    python script/compress_results.py --delete           # Delete original after compression
    python script/compress_results.py experiment/        # Compress only experiment/
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_subdirs_at_depth(root: Path, depth: int) -> List[Path]:
    """Get all subdirectories at a specific depth from root.

    Args:
        root: Root directory to start from
        depth: Depth level (1 = immediate children, 2 = grandchildren, etc.)

    Returns:
        List of directory paths at the specified depth
    """
    if not root.exists() or not root.is_dir():
        return []

    if depth <= 0:
        return [root] if root.is_dir() else []

    if depth == 1:
        return [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Recursively get subdirs at depth - 1 for each child
    result = []
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith('.'):
            result.extend(get_subdirs_at_depth(child, depth - 1))
    return result


def compress_directory(
    dir_path: Path,
    output_path: Optional[Path] = None,
    delete_original: bool = False,
    dry_run: bool = False,
    compression_level: int = 3,
) -> bool:
    """Compress a directory to a .tar.zst file.

    Args:
        dir_path: Directory to compress
        output_path: Output file path (default: {dir_path}.tar.zst)
        delete_original: Delete original directory after successful compression
        dry_run: Only print what would be done
        compression_level: zstd compression level (1-19, default 3)

    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = dir_path.parent / (dir_path.name + '.tar.zst')

    # Skip if already compressed
    if output_path.exists():
        print(f"  SKIP (exists): {output_path}")
        return True

    # Get directory size for reporting
    try:
        dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        dir_size_mb = dir_size / (1024 * 1024)
    except OSError:
        dir_size_mb = 0

    if dry_run:
        print(f"  WOULD COMPRESS: {dir_path} ({dir_size_mb:.1f} MB) -> {output_path.name}")
        return True

    print(f"  COMPRESSING: {dir_path} ({dir_size_mb:.1f} MB) -> {output_path.name}")

    # Use tar with zstd compression
    # tar -I 'zstd -3' -cf output.tar.zst -C parent_dir dir_name
    try:
        result = subprocess.run(
            [
                'tar',
                '-I', f'zstd -{compression_level}',
                '-cf', str(output_path),
                '-C', str(dir_path.parent),
                dir_path.name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Get compressed size
        compressed_size = output_path.stat().st_size
        compressed_size_mb = compressed_size / (1024 * 1024)
        ratio = (compressed_size / dir_size * 100) if dir_size > 0 else 0

        print(f"    -> {compressed_size_mb:.1f} MB ({ratio:.1f}% of original)")

        if delete_original:
            shutil.rmtree(dir_path)
            print(f"    -> Deleted original directory")

        return True

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to compress {dir_path}: {e.stderr}")
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        return False
    except FileNotFoundError:
        print("  ERROR: 'tar' or 'zstd' not found. Please install zstd.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress experiment results and bookkeeping directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'directories',
        nargs='*',
        default=['bookkeeping', 'experiment'],
        help="Directories to compress (default: bookkeeping experiment)",
    )

    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        help="Depth of subdirectories to compress (default: 2, e.g., experiment/bbq_age/exp_name)",
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be compressed without actually compressing",
    )

    parser.add_argument(
        '--delete',
        action='store_true',
        help="Delete original directories after successful compression",
    )

    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=range(1, 20),
        metavar='1-19',
        help="zstd compression level (default: 3, higher = smaller but slower)",
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(os.environ.get('MAC_FAIRNESS_WORKSPACE', '.')).resolve()

    # Collect all directories to compress
    dirs_to_compress: List[Path] = []

    for dir_name in args.directories:
        root = Path(dir_name)
        if not root.is_absolute():
            root = project_root / root

        if not root.exists():
            print(f"WARNING: Directory not found: {root}")
            continue

        subdirs = get_subdirs_at_depth(root, args.depth)
        dirs_to_compress.extend(subdirs)

    if not dirs_to_compress:
        print("No directories found to compress.")
        return 0

    # Sort for consistent ordering
    dirs_to_compress.sort()

    print(f"Found {len(dirs_to_compress)} directories to compress at depth {args.depth}")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)")
    print()

    # Compress each directory
    success_count = 0
    skip_count = 0
    fail_count = 0

    for dir_path in dirs_to_compress:
        # Make path relative for cleaner output
        try:
            rel_path = dir_path.relative_to(project_root)
        except ValueError:
            rel_path = dir_path

        output_path = dir_path.parent / (dir_path.name + '.tar.zst')

        if output_path.exists():
            skip_count += 1
            print(f"SKIP: {rel_path} (archive exists)")
            continue

        if compress_directory(
            dir_path,
            output_path=output_path,
            delete_original=args.delete,
            dry_run=args.dry_run,
            compression_level=args.level,
        ):
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"Summary: {success_count} compressed, {skip_count} skipped, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
