set(MNN_INC "" CACHE PATH "MNN SDK include directory.")
set(MNN_LIB "" CACHE PATH "MNN SDK library directory.")

find_library(MNN_LIB_DIR "MNN" PATHS ${MNN_LIB} NO_DEFAULT_PATH)
set(MNN_LIB ${MNN_LIB_DIR})
message("MNN_LIB = ${MNN_LIB}")

if(MNN_LIB)
  set(MNN_FOUND 1)
else()
  message(STATUS "DNN MNN backend: Failed to find libmnn in ${MNN_LIB}. Turning off MNN_FOUND!")
  message(STATUS "Please set the MNN_INC and MNN_LIB in the environment variable!")
  set(MNN_FOUND 0)
endif()

if(MNN_FOUND)
  set(HAVE_MNN 1)
endif()

if(HAVE_MNN)
  include_directories(${MNN_INC})
endif()

MARK_AS_ADVANCED(
  MNN_INC
  MNN_LIB
)
