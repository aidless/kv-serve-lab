# C++ 对拍 driver 的构建。
#
# 只依赖 g++（无第三方库、无 cmake —— 本机 cmake 不存在，AGENTS.md 里记过）。
# 产物固定落在 build/kvbm，tests/test_cpp_differential.py 按这个路径找。
#
# 目标用的是 .PHONY 形式而不是文件目标，理由是源文件只有一份：
# 与其维护头文件依赖，不如每次重编（编译 < 1 秒）。

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD := build
BIN := $(BUILD)/kvbm
SRC := kv_block_manager.cpp

.PHONY: all clean test

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC)

clean:
	rm -rf $(BUILD)

# 手工对拍：make -C cpp && build/kvbm 4 2 < 脚本
test: $(BIN)
	@echo "alloc 2" | $(BIN) 4 2
