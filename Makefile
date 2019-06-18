#
# $File: Makefile
# $Date: Fri Sep 12 22:51:24 2014 +0800
#
# A single output portable Makefile for
# simple c++ project


SRC_DIR = src
INC_DIR = src
OBJ_DIR = obj

TARGET1 = main1
TARGET2 = main2
TARGET3 = main3

CXX = $(ENVIRONMENT_OPTIONS) g++
BIN_TARGET1 = $(TARGET1)
BIN_TARGET2 = $(TARGET2)
BIN_TARGET3 = $(TARGET3)

INCLUDE_DIR = -I $(SRC_DIR) -I $(INC_DIR)

CXXFLAGS = -O2 -w
# CXXFLAGS = -O2 -w -fopenmp
# CXXFLAGS = -g
# CXXFLAGS = -pg

# CXXFLAGS += $(DEFINES)
CXXFLAGS += -std=c++17
# CXXFLAGS += -ansi
# CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(INCLUDE_DIR)
# CXXFLAGS += $(LDFLAGS)
# CXXFLAGS += -pthread -lpthread
# CXXFLAGS += -fPIC

CXXSOURCES = $(shell find $(SRC_DIR) -name *.cc)
CXXSOURCES_MAIN1 = examples/stage2/main.cc
CXXSOURCES_MAIN2 = examples/stage2/main.cc
CXXSOURCES_MAIN3 = examples/newton_method/main.cc

OBJS = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES:.cc=.o))
OBJS_MAIN1 = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES_MAIN1:.cc=.o))
OBJS_MAIN2 = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES_MAIN2:.cc=.o))
OBJS_MAIN3 = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES_MAIN3:.cc=.o))

OBJS_ALL = $(OBJS) $(OBJS_MAIN1) $(OBJS_MAIN2) $(OBJS_MAIN3)
DEPFILES_ALL = $(OBJS_ALL:.o=.d)

.PHONY: all clean run rebuild gdb

all: $(BIN_TARGET1) $(BIN_TARGET2) $(BIN_TARGET3)

$(OBJ_DIR)/%.d: %.cc
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(INCLUDE_DIR) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<" > "$@"

sinclude $(DEPFILES_ALL)

$(OBJ_DIR)/%.o: %.cc
	@echo "[CC] $< ..."
	@$(CXX) -c $< $(CXXFLAGS) -o $@

$(BIN_TARGET1): $(OBJS) $(OBJS_MAIN1)
	@echo $(OBJS) $(OBJS_MAIN1)
	@echo "[link] $< ..."
	@$(CXX) $(OBJS) $(OBJS_MAIN1) -o $@ $(CXXFLAGS)

$(BIN_TARGET2): $(OBJS) $(OBJS_MAIN2)
	@echo $(OBJS) $(OBJS_MAIN2)
	@echo "[link] $< ..."
	@$(CXX) $(OBJS) $(OBJS_MAIN2) -o $@ $(CXXFLAGS)

$(BIN_TARGET3): $(OBJS) $(OBJS_MAIN3)
	@echo $(OBJS) $(OBJS_MAIN3)
	@echo "[link] $< ..."
	@$(CXX) $(OBJS) $(OBJS_MAIN3) -o $@ $(CXXFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_TARGET1) $(BIN_TARGET2) $(BIN_TARGET3)

rebuild:
	+@make clean
	+@make

