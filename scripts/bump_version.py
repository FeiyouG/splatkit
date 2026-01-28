#!/usr/bin/env python3
"""
Helper script to bump version numbers consistently across files.

Usage:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py 1.0.0
    python scripts/bump_version.py 0.1.0a1
    python scripts/bump_version.py 0.2.0.dev1
    python scripts/bump_version.py 0.2.0.post1
"""

import sys
import re
from pathlib import Path

def update_pyproject_toml(version: str, root: Path):
    """Update version in pyproject.toml"""
    file_path = root / "pyproject.toml"
    content = file_path.read_text()
    
    # Update version line
    new_content = re.sub(
        r'version = "[^"]*"',
        f'version = "{version}"',
        content
    )
    
    file_path.write_text(new_content)
    print(f"✅ Updated {file_path}")

def update_init_py(version: str, root: Path):
    """Update version in splatkit/__init__.py"""
    file_path = root / "splatkit" / "__init__.py"
    content = file_path.read_text()
    
    # Update __version__ line
    new_content = re.sub(
        r'__version__ = "[^"]*"',
        f'__version__ = "{version}"',
        content
    )
    
    file_path.write_text(new_content)
    print(f"✅ Updated {file_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <version>")
        print("Example: python scripts/bump_version.py 0.2.0")
        sys.exit(1)
    
    version = sys.argv[1]
    
    # Validate version format (PEP 440 compliant)
    # Supports: release (X.Y.Z), pre-release (aN, bN, rcN), dev (.devN), post (.postN)
    if not re.match(r'^\d+\.\d+\.\d+(a\d+|b\d+|rc\d+|\.dev\d+|\.post\d+)?$', version):
        print(f"❌ Invalid version format: {version}")
        print("Expected PEP 440 format: X.Y.Z[a|b|rc|.dev|.post]N")
        print("Examples: 1.0.0, 1.0.0a1, 1.0.0b2, 1.0.0rc1, 1.0.0.dev1, 1.0.0.post1")
        sys.exit(1)
    
    root = Path(__file__).parent.parent
    
    print(f"Bumping version to {version}...")
    update_pyproject_toml(version, root)
    update_init_py(version, root)
    
    print(f"\n✨ Version bumped to {version}")
    print("\nNext steps:")
    print(f"  git add pyproject.toml splatkit/__init__.py")
    print(f"  git commit -m 'Bump version to {version}'")
    print(f"  git tag v{version}")
    print(f"  git push && git push --tags")

if __name__ == "__main__":
    main()
