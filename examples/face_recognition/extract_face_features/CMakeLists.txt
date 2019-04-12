cmake_minimum_required (VERSION 2.6)
project (vsc_pc)
set(CMAKE_CXX_FLAGS "-std=c++11 ")

set(CMAKE_MODULE_PATH ${vsc_pc_SOURCE_DIR})

#sqlite
include_directories(/usr/local/include)
link_directories(/usr/lib/x86_64-linux-gnu/)

find_package(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  MESSAGE("Found OpenCV")
  MESSAGE("OpenCV Includes: " ${OpenCV_INCLUDE_DIRS})
  MESSAGE("OpenCV Libraries directories: " ${OpenCV_LIBRARY_DIRS})
  MESSAGE("OpenCV Libraries: " ${OpenCV_LIBRARIES})
ENDIF(OpenCV_FOUND)

add_executable (main main.cpp sqlite_db.cpp)
target_link_libraries (main ${OpenCV_LIBRARIES} pthread sqlite3)
