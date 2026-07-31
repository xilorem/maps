PYTHON ?= ./.venv/bin/python
MAPS_IR_DIR ?= maps-ir

GENERATED_DIR := generated
EXECUTION_PLAN_JSON := $(GENERATED_DIR)/magia_example.execution-plan.json
EXECUTION_PLAN_WEIGHTS := $(basename $(EXECUTION_PLAN_JSON)).weights.bin
MODEL ?= examples/simple_three_stage.onnx
TARGET ?= magia-v2
MESH ?= 4
TOKEN_SLOTS ?= 2
PIPELINE_TOKEN_CAPACITY ?= 1
PACKAGE ?= $(GENERATED_DIR)/simple_three_stage.maps
MAPS_TRANSLATE ?= $(MAPS_IR_DIR)/build/tools/maps-translate/maps-translate

.PHONY: all package package-inspect package-verify magia-example magia-package-artifacts execution-plan-bundle execution-plan-json maps-translate maps-mlir magia-header magia-data clean-generated

all: magia-example

package:
	$(PYTHON) -m MAPS.cli package $(MODEL) \
		--target $(TARGET) \
		--mesh $(MESH) \
		--token-slots $(TOKEN_SLOTS) \
		--pipeline-token-capacity $(PIPELINE_TOKEN_CAPACITY) \
		--maps-translate $(MAPS_TRANSLATE) \
		--output $(PACKAGE)

package-inspect:
	$(PYTHON) -m MAPS.cli package inspect $(PACKAGE)

package-verify:
	$(PYTHON) -m MAPS.cli package verify $(PACKAGE)

magia-example: execution-plan-bundle
	$(MAKE) -C $(MAPS_IR_DIR) magia-example EXECUTION_PLAN_JSON=../$(EXECUTION_PLAN_JSON) GENERATED_DIR=../$(GENERATED_DIR)

magia-package-artifacts: execution-plan-bundle
	$(MAKE) -C $(MAPS_IR_DIR) magia-package \
		EXECUTION_PLAN_JSON=../$(EXECUTION_PLAN_JSON) \
		BUNDLE_WEIGHTS=../$(EXECUTION_PLAN_WEIGHTS) \
		MAGIA_PACKAGE_DIR=../$(GENERATED_DIR)/magia_example.runtime \
		OUTPUT_STEM=model

execution-plan-bundle:
	$(PYTHON) examples/magia_example.py
	test -f $(EXECUTION_PLAN_JSON)
	test -f $(EXECUTION_PLAN_WEIGHTS)

execution-plan-json: execution-plan-bundle

maps-translate:
	$(MAKE) -C $(MAPS_IR_DIR) maps-translate

maps-mlir: execution-plan-bundle
	$(MAKE) -C $(MAPS_IR_DIR) $@ EXECUTION_PLAN_JSON=../$(EXECUTION_PLAN_JSON) GENERATED_DIR=../$(GENERATED_DIR)

magia-header magia-data:
	if [ ! -f $(EXECUTION_PLAN_JSON) ]; then $(MAKE) execution-plan-bundle; fi
	$(MAKE) -C $(MAPS_IR_DIR) $@ EXECUTION_PLAN_JSON=../$(EXECUTION_PLAN_JSON) GENERATED_DIR=../$(GENERATED_DIR)

clean-generated:
	$(MAKE) -C $(MAPS_IR_DIR) clean-generated GENERATED_DIR=../$(GENERATED_DIR)
