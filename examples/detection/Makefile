####################### 
# Makefile 
####################### 

# target 
TARGET := Detect_Demo 

# compile and lib parameter 
CC := arm-hisiv300-linux-g++
LDFLAGS := ../../libs/
LIBS := SIMCAM_lib.a 
DEFINES := 
INCLUDE := -I ../../libs
CFLAGS := 
CXXFLAGS:= 

# link 
$(TARGET):Detect_Demo.o 
	$(CC) -o $@ $^ -lpthread $(LDFLAGS)$(LIBS)
	rm -fr *.o  
 
#compile 
Detect_Demo.o:Detect_Demo.c 
	$(CC) -c $^ -o $@ 
	
# clean 
clean: 
	rm -fr *.o 
	rm -fr $(TARGET)
