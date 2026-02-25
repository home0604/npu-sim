#!/bin/bash
# Setup script for DRAMSim3 integration
# This script clones DRAMSim3, builds it as a shared library,
# and compiles the pybind11 wrapper.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXT_DIR="$PROJECT_DIR/ext"

echo "=== Setting up DRAMSim3 ==="

# Clone DRAMSim3 if not present
if [ ! -d "$EXT_DIR/DRAMsim3" ]; then
    echo "Cloning DRAMSim3..."
    git clone https://github.com/umd-memsys/DRAMsim3.git "$EXT_DIR/DRAMsim3"
else
    echo "DRAMSim3 already cloned."
fi

cd "$EXT_DIR/DRAMsim3"

# Build DRAMSim3 as shared library
echo "Building DRAMSim3..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cd ..

echo "DRAMSim3 built successfully."

# Build pybind11 wrapper
echo "Building pybind11 wrapper..."

DRAMSIM3_DIR="$EXT_DIR/DRAMsim3"
WRAPPER_SRC="$EXT_DIR/dramsim3_wrapper.cpp"

# Create pybind11 wrapper source
cat > "$WRAPPER_SRC" << 'WRAPPER_EOF'
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <string>
#include <functional>
#include "dramsim3.h"

namespace py = pybind11;

PYBIND11_MODULE(dramsim3_py, m) {
    m.doc() = "DRAMSim3 Python bindings via pybind11";

    py::class_<dramsim3::MemorySystem>(m, "MemorySystem")
        .def(py::init([](const std::string& config_file,
                         const std::string& output_dir,
                         std::function<void(uint64_t)> read_cb,
                         std::function<void(uint64_t)> write_cb) {
            auto* mem = new dramsim3::MemorySystem(config_file, output_dir, read_cb, write_cb);
            return mem;
        }))
        .def("AddTransaction", &dramsim3::MemorySystem::AddTransaction,
             py::arg("addr"), py::arg("is_write"))
        .def("ClockTick", &dramsim3::MemorySystem::ClockTick)
        .def("PrintStats", &dramsim3::MemorySystem::PrintStats)
        .def("ResetStats", &dramsim3::MemorySystem::ResetStats);
}
WRAPPER_EOF

# Get pybind11 include paths
PYBIND11_INCLUDES=$(python3 -m pybind11 --includes 2>/dev/null || echo "")
if [ -z "$PYBIND11_INCLUDES" ]; then
    echo "WARNING: pybind11 not found. Install with: pip install pybind11"
    echo "Skipping wrapper build. SimpleDRAMModel will be used as fallback."
    exit 0
fi

# Get Python extension suffix
PY_EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null || echo ".so")

# Compile
DRAMSIM3_LIB="$DRAMSIM3_DIR/build"
DRAMSIM3_INC="$DRAMSIM3_DIR/src"

# Detect OS for correct flags
if [[ "$(uname)" == "Darwin" ]]; then
    SHARED_FLAG="-dynamiclib -undefined dynamic_lookup"
else
    SHARED_FLAG="-shared"
fi

g++ -O2 -std=c++17 $SHARED_FLAG -fPIC \
    $PYBIND11_INCLUDES \
    -I"$DRAMSIM3_INC" \
    -L"$DRAMSIM3_LIB" \
    "$WRAPPER_SRC" \
    -ldramsim3 \
    -o "$PROJECT_DIR/dramsim3_py${PY_EXT_SUFFIX}" \
    -Wl,-rpath,"$DRAMSIM3_LIB"

echo "pybind11 wrapper built: $PROJECT_DIR/dramsim3_py${PY_EXT_SUFFIX}"
echo ""
echo "=== Setup complete ==="
echo "Add $PROJECT_DIR to PYTHONPATH or install the package to use dramsim3_py."
