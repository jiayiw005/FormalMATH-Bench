#!/bin/bash
# ============================================================================
# OSCAR Setup: Lean REPL + Mathlib for Grind Verification
#
# Run on OSCAR login node (or an interactive session):
#   bash scripts/oscar_lean_setup.sh
#
# Everything is installed under $SCRATCH — nothing touches $HOME.
# ============================================================================

set -euo pipefail

export SCRATCH_DIR="/users/${USER}/scratch"
export PROJECT_DIR="${SCRATCH_DIR}/grind_project"

# ── Force elan/lean/lake to live entirely on scratch ──
export ELAN_HOME="${SCRATCH_DIR}/.elan"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export PATH="${ELAN_HOME}/bin:${PATH}"

echo "============================================"
echo "  Lean REPL + Mathlib Setup on OSCAR"
echo "============================================"
echo "  User:        ${USER}"
echo "  Project:     ${PROJECT_DIR}"
echo "  ELAN_HOME:   ${ELAN_HOME}"
echo ""

# ---------------------------------------------------------------------------
# 1. Install elan (Lean version manager) into scratch
# ---------------------------------------------------------------------------
echo "[1/5] Installing elan into ${ELAN_HOME}..."

if [ -f "${ELAN_HOME}/bin/elan" ]; then
    echo "  elan already installed."
else
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- \
        --default-toolchain none \
        --no-modify-path \
        -y
    echo "  elan installed."
fi

echo "  elan version: $(elan --version)"
echo ""

# ---------------------------------------------------------------------------
# 2. Clone the REPL from GitHub
# ---------------------------------------------------------------------------
echo "[2/6] Cloning Lean REPL..."
if [ -d "${PROJECT_DIR}/repl/REPL" ]; then
    echo "  REPL source already present, skipping clone."
else
    # Clone into a temp dir, then move contents into repl/
    REPL_TMP=$(mktemp -d)
    git clone --depth 1 --branch v4.28.0-rc1 https://github.com/leanprover-community/repl.git "${REPL_TMP}"
    # Preserve any existing repl/ contents (e.g. test/Mathlib/lakefile.lean)
    mkdir -p "${PROJECT_DIR}/repl"
    cp -rn "${REPL_TMP}"/* "${PROJECT_DIR}/repl/" 2>/dev/null || true
    cp -rn "${REPL_TMP}"/.[!.]* "${PROJECT_DIR}/repl/" 2>/dev/null || true
    rm -rf "${REPL_TMP}"
    echo "  Cloned leanprover-community/repl into ${PROJECT_DIR}/repl/"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Install the correct Lean toolchain
# ---------------------------------------------------------------------------
LEAN_VERSION=$(cat "${PROJECT_DIR}/repl/lean-toolchain" | tr -d '[:space:]')
echo "[3/6] Installing Lean toolchain: ${LEAN_VERSION}..."

elan toolchain install "${LEAN_VERSION}" || true
elan default "${LEAN_VERSION}"
echo "  lean version: $(lean --version)"
echo ""

# ---------------------------------------------------------------------------
# 4. Build the REPL
# ---------------------------------------------------------------------------
echo "[4/6] Building Lean REPL..."
cd "${PROJECT_DIR}/repl"

lake build repl
echo "  REPL binary: $(ls -lh .lake/build/bin/repl)"
echo ""

# ---------------------------------------------------------------------------
# 5. Get Mathlib and download oleans cache
# ---------------------------------------------------------------------------
echo "[5/6] Setting up Mathlib environment..."
cd "${PROJECT_DIR}/repl/test/Mathlib"

# Redirect cache to scratch (avoid filling $HOME)
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

# OSCAR's system curl is "too old" for Mathlib, which then downloads its own
# curl that is INCOMPATIBLE with OSCAR's OpenSSL 3.0.8. Fix: install a
# static curl binary (with bundled SSL) and put it first in PATH so Mathlib
# never needs to download its own.
CURL_DIR="${SCRATCH_DIR}/bin"
mkdir -p "${CURL_DIR}"
if [ ! -f "${CURL_DIR}/curl" ]; then
    echo "  Downloading static curl binary..."
    /usr/bin/curl -sSL -o "${CURL_DIR}/curl" \
        https://github.com/moparisthebest/static-curl/releases/latest/download/curl-amd64
    chmod +x "${CURL_DIR}/curl"
    echo "  Static curl installed: $(${CURL_DIR}/curl --version | head -1)"
else
    echo "  Static curl already installed."
fi
export PATH="${CURL_DIR}:${PATH}"

# Static curl expects Debian CA path; OSCAR (RHEL) uses a different location
if [ -f /etc/pki/tls/certs/ca-bundle.crt ]; then
    export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
elif [ -f /etc/ssl/cert.pem ]; then
    export CURL_CA_BUNDLE=/etc/ssl/cert.pem
fi

# Remove any previously cached bad curl from Mathlib
rm -rf "${XDG_CACHE_HOME}/mathlib/curl-"* 2>/dev/null || true

echo "  Running lake update (fetching Mathlib + dependencies)..."
lake update

echo "  Downloading Mathlib oleans cache..."
lake exe cache get

echo "  Mathlib cache downloaded."
echo ""

# ---------------------------------------------------------------------------
# 6. Build Mathlib (compiles anything not covered by cache)
# ---------------------------------------------------------------------------
echo "[6/6] Building Mathlib (this may take a while if cache is incomplete)..."
# lake build

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  REPL binary:     ${PROJECT_DIR}/repl/.lake/build/bin/repl"
echo "  Mathlib env:     ${PROJECT_DIR}/repl/test/Mathlib"
echo ""
# Add elan to ~/.bashrc if not already there
if ! grep -q 'ELAN_HOME.*\.elan' ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Lean / elan (added by oscar_lean_setup.sh)" >> ~/.bashrc
    echo "export ELAN_HOME=\"${ELAN_HOME}\"" >> ~/.bashrc
    echo 'export PATH="${ELAN_HOME}/bin:${PATH}"' >> ~/.bashrc
    echo "  Added ELAN_HOME + PATH to ~/.bashrc"
else
    echo "  ~/.bashrc already has ELAN_HOME configured"
fi
echo ""
echo "  Run verify scripts with:"
echo "    python3 verify_grind_replace.py \\"
echo "      --repl_path ${PROJECT_DIR}/repl \\"
echo "      --lean_env_path ${PROJECT_DIR}/repl/test/Mathlib \\"
echo "      --input data/goedel_sft/proofs_top30k.jsonl \\"
echo "      --output data/goedel_sft/grind_replace_results_top30k.jsonl"
