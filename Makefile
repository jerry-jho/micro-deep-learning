MY:=$(shell pwd)/$(lastword $(MAKEFILE_LIST))
UDL_REL:=$(shell dirname $(MY))
UDL:=$(shell realpath $(UDL_REL))
SRC:=$(UDL)/src
TARGETS:=os generic
BUILD:=$(PWD)/build
CROSS:=
CROSS_CFLAGS:=
CROSS_LDFLAGS:=

SOURCE:=$(shell bash $(UDL)/tools/get_file_list.sh $(SRC) $(TARGETS))
TEST_SOURCES:=$(wildcard $(PWD)/*.c)

INCLUDES:=-I$(SRC)/generic

define compile_object
$(1):$(2)
	$(CROSS)gcc -Wall $(CROSS_CFLAGS) -c $(INCLUDES) $(2) -o $(1)
endef

define get_base
$(basename $(notdir $(1)))
endef

OBJS:=$(foreach name,$(SOURCE),$(BUILD)/$(call get_base,$(name)).o)

$(foreach name,$(SOURCE),\
	$(eval $(call compile_object,$(BUILD)/$(call get_base,$(name)).o,$(name)))\
)

$(BUILD):
	mkdir -p $(BUILD)

$(TEST_SOURCES) : $(BUILD)

test: $(TEST_SOURCES) $(OBJS)
	$(CROSS)gcc -Wall $(INCLUDES) $(CROSS_CFLAGS) $(CROSS_LDFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD)
