#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Force Apple toolchain to avoid libc++ mismatch issues on macOS.
export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"

# Clean stale outputs that can shadow imports.
rm -rf build ar_native.egg-info ar_native*.so native/*.o native/*.obj 2>/dev/null || true

python -m pip install -U pip setuptools wheel >/dev/null
python -m pip install -e .

python -c "import ar_native; import numpy as np; a=np.zeros((4,4,3),dtype=np.uint8); M=np.eye(3); b=ar_native.warp_perspective_u8(a,M,4,4); print('ar_native OK:', b.shape, b.dtype)"

echo "Built and verified ar_native."