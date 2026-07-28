PYTHON ?= ./.venv/bin/python
MAPS_IR_DIR ?= maps-ir

GENERATED_DIR := generated
PIPELINE_JSON := $(GENERATED_DIR)/magia_example.pipeline.json
PIPELINE_WEIGHTS := $(basename $(PIPELINE_JSON)).weights.bin
MODEL ?= examples/simple_three_stage.onnx
TARGET ?= magia-v2
MESH ?= 4
TOKEN_SLOTS ?= 2
PIPELINE_TOKEN_CAPACITY ?= 1
PACKAGE ?= $(GENERATED_DIR)/simple_three_stage.maps
MAPS_TRANSLATE ?= $(MAPS_IR_DIR)/build/tools/maps-translate/maps-translate

.PHONY: all package package-inspect package-verify magia-example magia-package-artifacts pipeline-bundle pipeline-json maps-translate maps-mlir magia-header magia-data clean-generated

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

magia-example: pipeline-bundle
	$(MAKE) -C $(MAPS_IR_DIR) magia-example PIPELINE_JSON=../$(PIPELINE_JSON) GENERATED_DIR=../$(GENERATED_DIR)

magia-package-artifacts: pipeline-bundle
	$(MAKE) -C $(MAPS_IR_DIR) magia-package \
		PIPELINE_JSON=../$(PIPELINE_JSON) \
		BUNDLE_WEIGHTS=../$(PIPELINE_WEIGHTS) \
		MAGIA_PACKAGE_DIR=../$(GENERATED_DIR)/magia_example.runtime \
		OUTPUT_STEM=model

pipeline-bundle:
	$(PYTHON) examples/magia_example.py
	test -f $(PIPELINE_JSON)
	test -f $(PIPELINE_WEIGHTS)

pipeline-json: pipeline-bundle

maps-translate:
	$(MAKE) -C $(MAPS_IR_DIR) maps-translate

maps-mlir: pipeline-bundle
	$(MAKE) -C $(MAPS_IR_DIR) $@ PIPELINE_JSON=../$(PIPELINE_JSON) GENERATED_DIR=../$(GENERATED_DIR)

magia-header magia-data:
	if [ ! -f $(PIPELINE_JSON) ]; then $(MAKE) pipeline-bundle; fi
	$(MAKE) -C $(MAPS_IR_DIR) $@ PIPELINE_JSON=../$(PIPELINE_JSON) GENERATED_DIR=../$(GENERATED_DIR)

clean-generated:
	$(MAKE) -C $(MAPS_IR_DIR) clean-generated GENERATED_DIR=../$(GENERATED_DIR)
