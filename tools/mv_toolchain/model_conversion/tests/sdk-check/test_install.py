import sys
if sys.version_info[0] != 3:
    print("Incompatible Version of Python. Please use Python 3.")
    quit()


import platform
if platform.system() != "Linux" or platform.dist()[0] != "Ubuntu" or platform.dist()[1] != "16.04":
	if platform.dist()[0] != "debian" or platform.dist()[1] != "9.1":
		print("Warning: Potentially unsupported OS")

def custom_import_function(module, error_msg):
    try:
        __import__(module)
    except Exception:
        print(error_msg)

print("Performing NCSDK python3 dependencies check...")

custom_import_function('os', "Python STD Lib problem")
custom_import_function('sys', "Python STD Lib problem")
custom_import_function('argparse', "Python STD Lib problem")
custom_import_function('argparse', "Python STD Lib problem")
custom_import_function('ctypes', "Python STD Lib problem")
custom_import_function('math', "Python STD Lib problem")
custom_import_function('re', "Python STD Lib problem")
custom_import_function('warnings', "Python STD Lib problem")
custom_import_function('struct', "Python STD Lib problem")

custom_import_function('datetime', "DateTime Python Libary not found.")
custom_import_function('numpy', "Numpy Python Libary not found.")
custom_import_function('yaml', "YAML Python Libary not found.")
custom_import_function('google.protobuf', "Protobuf Python Libary not found.")
custom_import_function('caffe', "Caffe Python Libary: import error #1. Import not successful")


try:
	import numpy
	version = numpy.version.version.split(".")
	assert(int(version[0]) >= 1)
	if version[0] == 1:
		assert(int(version[1]) >= 11)
	
except Exception:
    print ("Warning: Numpy must be at least 1.11.0")

try:
    from enum import Enum
except Exception:
    print ("Python problem. Std libraries not found.")

try:
    from csv import writer
except Exception:
    print ("Python problem. CSV library not found.")


try:
    from caffe.proto import caffe_pb2
except ImportError:
    print("Caffe Python Libary Import Error #2: import caffe_pb2")

try:
    from graphviz import Digraph
except Exception:
    print("graphviz Python Libary not found")

print("NCSDK python3 dependencies check passed!")
