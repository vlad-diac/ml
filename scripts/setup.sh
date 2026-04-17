#!/bin/bash
# DEPRECATED — use scripts/install.sh instead.
#
# scripts/install.sh is the new setup wizard. It:
#   - Detects your OS and architecture (macOS arm64/x86_64, Linux x86_64/aarch64)
#   - Installs Miniforge3 (conda) if not present
#   - Creates the "sky-seg" conda environment with Python 3.11
#   - Installs the correct TensorFlow variant for your platform
#   - Installs all remaining project dependencies
#
# Run:
#   bash scripts/install.sh

echo ""
echo "  *** DEPRECATED ***"
echo "  setup.sh has been replaced by scripts/install.sh."
echo ""
echo "  Run the new setup wizard instead:"
echo "    bash scripts/install.sh"
echo ""
exit 1
