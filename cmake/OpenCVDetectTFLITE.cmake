set(TFLITE_INC "" CACHE PATH "TFLITE SDK include directory.")
set(TFLITE_GEMMLOWP_INC "" CACHE PATH "GEMMLOWP include directory.")
set(TFLITE_FLATBUFFERS_INC "" CACHE PATH "FLATBUFFERS include directory.")
set(TFLITE_LIB "" CACHE PATH "TFLITE SDK library directory.")

# TODO need to remove the following code for android bug at toolchain.cmake
set(CMAKE_FIND_ROOT_PATH ${TFLITE_LIB})
find_library(TFLITE_LIB_DIR NAMES "tensorflow-lite" PATHS ${TFLITE_LIB} NO_DEFAULT_PATH)
message(STATUS "finding tensorflow-lite at ${TFLITE_LIB_DIR}")
set(TFLITE_LIB ${TFLITE_LIB_DIR})

if(TFLITE_LIB)
  set(TFLITE_FOUND 1)
  message(STATUS "Found TFLITE from given path! TFLITE_LIB = ${TFLITE_LIB}")
else()
  message(STATUS "Not found tensorflow-lite at ${TFLITE_LIB}, set it to OFF!")
endif()

if(TFLITE_FOUND)
  set(HAVE_TFLITE 1)
endif()


MARK_AS_ADVANCED(
        TFLITE_INC
        TFLITE_LIB
        TFLITE_FLATBUFFERS_INC
        TFLITE_GEMMLOWP_INC
)
