PYTHON ?= ./.venv/bin/python
MAPS_IR_DIR ?= maps-ir

BUILD_DIR ?= build
MODEL ?= examples/simple_three_stage.onnx
TARGET ?= magia-v2
MESH ?=
MESH_OPTION = $(if $(strip $(MESH)),--mesh $(MESH))
TOKEN_SLOTS ?= 2
NAME ?=
NAME_OPTION = $(if $(strip $(NAME)),--name $(NAME))
INPUT ?=
INPUT_OPTION = $(if $(strip $(INPUT)),--input $(INPUT))
MAX_STAGE_OPERATIONS ?= 0
EXECUTION_PLAN ?=
EXECUTION_PLAN_OPTION = $(if $(strip $(EXECUTION_PLAN)),--output $(EXECUTION_PLAN))
APPLICATION ?= $(BUILD_DIR)/application

.PHONY: all test build plan inspect verify maps-ir clean-generated

all: test

test:
	$(PYTHON) -m pytest -q

plan:
	$(PYTHON) -m maps.cli plan $(MODEL) \
		--target $(TARGET) \
		$(MESH_OPTION) \
		--token-slots $(TOKEN_SLOTS) \
		--max-stage-operations $(MAX_STAGE_OPERATIONS) \
		$(EXECUTION_PLAN_OPTION)

build:
	$(PYTHON) -m maps.cli build $(MODEL) \
		--target $(TARGET) \
		$(MESH_OPTION) \
		--token-slots $(TOKEN_SLOTS) \
		$(NAME_OPTION) \
		$(INPUT_OPTION) \
		--output $(APPLICATION)

inspect:
	$(PYTHON) -m maps.cli inspect $(APPLICATION)

verify:
	$(PYTHON) -m maps.cli verify $(APPLICATION)

maps-ir:
	$(MAKE) -C $(MAPS_IR_DIR) tools

clean-generated:
	$(MAKE) -C $(MAPS_IR_DIR) clean-generated GENERATED_DIR=../$(BUILD_DIR)
