####################### 
# Makefile 
####################### 

# target 
TARGET := Demo 

# compile and lib parameter 
CC := arm-hisiv300-linux-g++
SQLITE_LIB := -L ../../libs/sqlite_lib -lsqlite3
SIMCAM_lib:= ../../libs/SIMCAM_lib.a
LIBS := $(SIMCAM_lib) $(SQLITE_LIB)
DEFINES := 
INCLUDE := -I ../../libs
CFLAGS := 
CXXFLAGS:= 

# link 
$(TARGET):Demo.o  sqlite_db.o
	$(CC) -o $@ $^ -lpthread $(LIBS) 
	rm -fr *.o  
 
#compile 


Demo.o:Demo.c 
		$(CC) -c $^ -o $@ 

sqlite_db.o: extract_face_features/sqlite_db.cpp
			$(CC) -c extract_face_features/sqlite_db.cpp


# clean 
clean: 
	rm -fr *.o 
	rm -fr $(TARGET)
