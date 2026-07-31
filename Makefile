PYTHON ?= ./.venv/bin/python
MAPS_IR_DIR ?= maps-ir

GENERATED_DIR ?= generated
MODEL ?= examples/simple_three_stage.onnx
TARGET ?= magia
MESH ?= $(if $(filter n300d,$(TARGET)),8x8,4x4)
TOKEN_SLOTS ?= 2
MAX_STAGE_NODES ?= 0
PIPELINE_TOKEN_CAPACITY ?= 1
EXECUTION_PLAN ?= $(GENERATED_DIR)/$(TARGET).execution-plan.json
PACKAGE ?= $(GENERATED_DIR)/$(TARGET).maps
MAPS_TRANSLATE ?= $(MAPS_IR_DIR)/build/tools/maps-translate/maps-translate

.PHONY: all test plan package inspect verify maps-translate clean-generated

all: test

test:
	$(PYTHON) -m pytest -q

plan:
	$(PYTHON) -m maps.cli plan $(MODEL) \
		--target $(TARGET) \
		--mesh $(MESH) \
		--token-slots $(TOKEN_SLOTS) \
		--max-stage-nodes $(MAX_STAGE_NODES) \
		--output $(EXECUTION_PLAN)

package:
	$(PYTHON) -m maps.cli package $(MODEL) \
		--target $(TARGET) \
		--mesh $(MESH) \
		--token-slots $(TOKEN_SLOTS) \
		--pipeline-token-capacity $(PIPELINE_TOKEN_CAPACITY) \
		--maps-translate $(MAPS_TRANSLATE) \
		--output $(PACKAGE)

inspect:
	$(PYTHON) -m maps.cli package inspect $(PACKAGE)

verify:
	$(PYTHON) -m maps.cli package verify $(PACKAGE)

maps-translate:
	$(MAKE) -C $(MAPS_IR_DIR) maps-translate

clean-generated:
	$(MAKE) -C $(MAPS_IR_DIR) clean-generated GENERATED_DIR=../$(GENERATED_DIR)
