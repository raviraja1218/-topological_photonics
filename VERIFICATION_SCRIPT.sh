#!/bin/bash
echo "========================================="
echo "TOPOLOGICAL PHOTONICS NATURE PAPER"
echo "PHASE 1 VERIFICATION"
echo "========================================="

# Set color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Counter for passed tests
PASSED=0
TOTAL=0

# 1. Check directory structure
echo -e "\n📁 Checking directory structure..."
TOTAL=$((TOTAL+1))
if [ -d "src/topological_pinn" ] && [ -d "src/inverse_design" ] && [ -d "data/topological" ] && [ -d "results" ]; then
    echo -e "${GREEN}✓${NC} Directory structure complete"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Missing directories"
fi

# 2. Check Conda environment
TOTAL=$((TOTAL+1))
if conda env list | grep -q "topological_photonics"; then
    echo -e "${GREEN}✓${NC} Conda environment 'topological_photonics' exists"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Conda environment missing"
fi

# 3. Check active environment
TOTAL=$((TOTAL+1))
if [[ "$CONDA_DEFAULT_ENV" == "topological_photonics" ]]; then
    echo -e "${GREEN}✓${NC} Environment active"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Wrong environment active (current: $CONDA_DEFAULT_ENV)"
fi

# 4. Check DeepXDE
TOTAL=$((TOTAL+1))
python -c "import deepxde as dde; print(f'   DeepXDE version: {dde.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} DeepXDE installed"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} DeepXDE not installed"
fi

# 5. Check MEEP
TOTAL=$((TOTAL+1))
python -c "import meep as mp; print(f'   MEEP version: {mp.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} MEEP installed"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} MEEP not installed"
fi

# 6. Check PyTorch
TOTAL=$((TOTAL+1))
python -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch installed"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} PyTorch not installed"
fi

# 7. Check configuration files
TOTAL=$((TOTAL+1))
if [ -f "config/moire_params.yaml" ] && [ -f "config/pinn_config.yaml" ] && [ -f "config/vae_config.yaml" ] && [ -f "config/ga_config.yaml" ] && [ -f "config/quantum_limits.yaml" ]; then
    echo -e "${GREEN}✓${NC} All configuration files present"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Missing configuration files"
fi

# 8. Check environment files
TOTAL=$((TOTAL+1))
if [ -f "environment.yml" ] && [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✓${NC} Environment files created"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Missing environment files"
fi

# 9. Check disk space
TOTAL=$((TOTAL+1))
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
if (( $(echo "$AVAILABLE > 20" | bc -l) )); then
    echo -e "${GREEN}✓${NC} Sufficient disk space: ${AVAILABLE}GB"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Low disk space: ${AVAILABLE}GB"
fi

# 10. Check Python version
TOTAL=$((TOTAL+1))
PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_VERSION" == "3.9" ]]; then
    echo -e "${GREEN}✓${NC} Python $PY_VERSION (correct)"
    PASSED=$((PASSED+1))
else
    echo -e "${RED}✗${NC} Python $PY_VERSION (need 3.9)"
fi

# Summary
echo -e "\n========================================="
echo -e "RESULTS: ${PASSED}/${TOTAL} checks passed"
echo -e "========================================="

if [ $PASSED -eq $TOTAL ]; then
    echo -e "${GREEN}✅ PHASE 1 COMPLETE - Ready for Phase 2${NC}"
    echo "PHASE 1 COMPLETED: $(date)" > PHASE1_COMPLETE.txt
    echo "PASSED: ${PASSED}/${TOTAL}" >> PHASE1_COMPLETE.txt
else
    echo -e "${RED}❌ PHASE 1 INCOMPLETE - Fix issues before Phase 2${NC}"
    exit 1
fi
