# thin driver for the flexdecomp framework (see AGENTS.md)
SHELL := /bin/bash
export PATH := /opt/cuda/bin:$(PATH)
KT := bin/flexkit

.PHONY: all setup inventory extract dwarf headers decomp stubs link units gates calib report build clean
all: setup build

setup: inventory extract dwarf headers stubs link units
	@echo "workspace ready"

inventory:
	$(KT) inventory

extract:
	$(KT) extract

dwarf:
	$(KT) dwarf

headers:
	$(KT) headers

decomp:
	$(KT) decomp

stubs:
	$(KT) stubs

link:
	$(KT) link --variant rel
	$(KT) link --variant dbg

units:
	$(KT) units build

build:
	ninja -C recon/build

gates: build
	$(KT) gate static --all
	$(KT) gate diff --all
	$(KT) gate gpu --all

calib: build
	$(KT) calib

report:
	$(KT) report

clean:
	rm -rf work/binary work/gen recon/build
