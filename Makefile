# ==============================
# Project configuration
# ==============================
ROOT_DIR := $(shell pwd)
SRC_DIR := src
OBJ_DIR := obj
OUT_DIR := bin

# Executables
TARGETS := inference_driver instrumentation_harness pipelined_inference_driver

# Compiler & flags
CXX := g++
CXXFLAGS := -std=c++17 -O3 -w
LDFLAGS := -Wl,-rpath=\$$ORIGIN/../lib \
	-lpthread \
	-ltensorflowlite \
	-ltensorflowlite_gpu_delegate -lEGL -lGLESv2 \
	-lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
	-ljsoncpp

# Include directories
INCS := -Iinc \
	-I$(SRC_DIR) \
	-I/usr/include \
	-I/usr/include/opencv4 \
	-I$(ROOT_DIR)/external/litert

# Library search paths
LIBS := -Llib

# ==============================
# Source groups
# ==============================
COMMON_SRCS := util.cpp
INFERENCE_DRIVER_SRCS := inference_driver.cpp $(COMMON_SRCS)
INST_HARNESS_SRCS := instrumentation_harness.cpp instrumentation_harness_utils.cpp $(COMMON_SRCS)
PIPELINED_DRIVER_SRCS := pipelined_inference_driver.cpp $(COMMON_SRCS)

# Map sources to objects
INFERENCE_DRIVER_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(INFERENCE_DRIVER_SRCS))
INST_HARNESS_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(INST_HARNESS_SRCS))
PIPELINED_DRIVER_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(PIPELINED_DRIVER_SRCS))

# Final binaries
INFERENCE_DRIVER_BIN := $(OUT_DIR)/inference_driver
INST_HARNESS_BIN := $(OUT_DIR)/instrumentation_harness
PIPELINED_DRIVER_BIN := $(OUT_DIR)/pipelined_inference_driver

# ==============================
# Build targets
# ==============================
.PHONY: all clean

# Build all if no option is provided
all: $(INFERENCE_DRIVER_BIN) $(INST_HARNESS_BIN)
	@echo "Build completed successfully."

# Optional single target build
inference: $(INFERENCE_DRIVER_BIN)
internals: $(INST_HARNESS_BIN)
pipelined: $(PIPELINED_DRIVER_BIN)

# Build rules for each binary
$(INFERENCE_DRIVER_BIN): $(INFERENCE_DRIVER_OBJS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) $^ -o $@ $(LDFLAGS)

$(INST_HARNESS_BIN): $(INST_HARNESS_OBJS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) $^ -o $@ $(LDFLAGS)

$(PIPELINED_DRIVER_BIN): $(PIPELINED_DRIVER_OBJS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) $^ -o $@ $(LDFLAGS)

# Compile objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(OUT_DIR)
