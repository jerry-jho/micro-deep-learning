MY:=$(shell pwd)/$(lastword $(MAKEFILE_LIST))
UDL_REL:=$(shell dirname $(MY))
UDL:=$(shell realpath $(UDL_REL))
SRC:=$(UDL)/src
TARGETS:=os generic
BUILD:=$(PWD)/build
CROSS:=
CROSS_CFLAGS:=
CROSS_LDFLAGS:=
DATA_FILE:=
CFLAGS:=-Wall -Wno-strict-aliasing


SOURCE:=$(shell bash $(UDL)/tools/get_file_list.sh $(SRC) $(TARGETS))
TEST_SOURCES:=$(wildcard $(PWD)/*.c)

INCLUDES:=-I$(SRC)/generic

define compile_object
$(1):$(2)
	$(CROSS)gcc $(CFLAGS) $(CROSS_CFLAGS) -c $(INCLUDES) $(2) -o $(1)
endef

ifneq ("$(wildcard $(DATA_FILE))","")
DATA_FILE_OBJ=$(BUILD)/_data.o
$(eval $(call compile_object,$(DATA_FILE_OBJ),$(DATA_FILE)))
else
DATA_FILE_OBJ=
endif

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

test: $(TEST_SOURCES) $(OBJS) $(DATA_FILE_OBJ)
	$(CROSS)gcc $(CFLAGS) $(INCLUDES) $(CROSS_CFLAGS) $(CROSS_LDFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD)
