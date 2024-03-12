set(TFLITE_INC "" CACHE PATH "TFLITE SDK include directory.")
set(TFLITE_LIB "" CACHE PATH "TFLITE SDK library directory.")

find_library(TFLITE_LIB_DIR "tensorflow-lite" PATHS ${TFLITE_LIB} NO_DEFAULT_PATH)
set(TFLITE_LIB ${TFLITE_LIB_DIR})

if(TFLITE_LIB)
  set(TFLITE_FOUND 1)
  message(STATUS "Found TFLITE from given path! TFLITE_LIB = ${TFLITE_LIB}")
else()
  message(STATUS "Not found tensorflow-lite, set it to OFF!")
endif()

if(TFLITE_FOUND)
  set(HAVE_TFLITE 1)
endif()

if(HAVE_TFLITE)
  include_directories(${TFLITE_INC})
endif()

MARK_AS_ADVANCED(
        TFLITE_INC
        TFLITE_LIB
)
