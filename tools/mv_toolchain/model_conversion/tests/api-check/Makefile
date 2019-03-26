CFLAGS := -O2 -Wall

all: ncs-check ncs-c-check ncs-python3-check ncs-python2-check 

objs := $(patsubst %.c,%.o,$(wildcard *.c))


ncs-check: $(objs)
	$(CC) $(objs) -lusb-1.0 -lmvnc -o $@

%.o: %c
	$(CC) -c $(CFLAGS) $<

.PHONY: ncs-c-check
ncs-c-check:
	@echo "****************************************************************"
	@echo "Checking NCS C API..."
	@echo "****************************************************************"
	./ncs-check -1

.PHONY: ncs-python3-check
ncs-python3-check:
	@echo "****************************************************************"
	@echo "Checking NCS python3 API..."
	@echo "****************************************************************"
	python3 ncs-python3-check.py

.PHONY: ncs-python2-check
ncs-python2-check:
	@echo "****************************************************************"
	@echo "Checking NCS python2 API..."
	@echo "****************************************************************"
	python ncs-python2-check.py

clean:
	rm -f *.o ncs-check
