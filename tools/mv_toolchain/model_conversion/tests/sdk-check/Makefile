python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_minor := $(word 2,${python_version_full})

all: check-dependencies check-compiler check-profiler check-checker

.PHONY: py
py:
	@echo ${python_version_minor}


.PHONY: check-dependencies
check-dependencies:
	@echo "****************************************************************"
	@echo "Checking NCS python dependencies..."
	@echo "****************************************************************"
	@python3 test_install.py

.PHONY: help
help:
	@cat Readme.txt
	@grep "echo" Makefile | egrep -v "grep|ERR" | sed 's/.*echo/   make/g'  | sed 's/\"//g'

.PHONY: check-compiler
check-compiler:
	@echo "****************************************************************"
	@echo "Checking NCS compiler..."
	@echo "****************************************************************"
	mvNCCompile data/googlenet.prototxt -s 12

.PHONY: check-profiler
check-profiler:
	@echo ""
	@echo "****************************************************************"
	@echo "Checking NCS profiler..."
	@echo "****************************************************************"
	mvNCProfile data/googlenet.prototxt -s 12

.PHONY: check-checker
check-checker:
	@echo ""
	@echo "****************************************************************"
	@echo "Checking the NCS graph checker..."
	@echo "****************************************************************"
	mvNCCheck data/googlenet.prototxt -i img/512_Phone.jpg -s 12

.PHONY: clean
clean:
	rm output* graph zero_weights*
