set(ORT_SDK "" CACHE PATH "ONNXRuntime SDK directory.")

find_library(ORT_SDK_LIB "onnxruntime" PATHS "${ORT_SDK}/lib" NO_DEFAULT_PATH)
set(ORT_SDK_INC "${ORT_SDK}/include")

if(ORT_SDK_LIB)
  set(ORT_FOUND 1)
else()
  message(STATUS "DNN ONNXRuntime backend: Failed to find libonnxruntime in ${ORT_SDK}/lib. Turning off ONNXRuntime_FOUND!")
  message(STATUS "Please set the ONNXRuntime_SDK in the environment variable!")
  set(ORT_FOUND 0)
endif()

if(ORT_FOUND)
  set(HAVE_ORT 1)
endif()

if(HAVE_ORT)
  include_directories(${ORT_SDK_INC})
endif()

MARK_AS_ADVANCED(
  ORT_SDK_INC
  ORT_SDK_LIB
)
