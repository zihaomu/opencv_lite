set(MNN_INC "" CACHE PATH "MNN SDK include directory.")
set(MNN_LIB "" CACHE PATH "MNN SDK library directory.")

find_library(MNN_LIB_DIR "MNN" PATHS ${MNN_LIB} NO_DEFAULT_PATH)
set(MNN_LIB ${MNN_LIB_DIR})

if(MNN_LIB)
  set(MNN_FOUND 1)
  message(STATUS "Found MNN from given path! MNN_LIB = ${MNN_LIB}")
else() # try to build MNN from source code.

  message(STATUS "Start to build MNN from source code...")
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/MNN")
  message(STATUS "Finish to build MNN from source code!")
  ocv_install_target(MNN EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)

  if (TARGET MNN)
    set(MNN_FOUND 1)
    set(MNN_LIB MNN)
    set(MNN_INC ${OpenCV_SOURCE_DIR}/3rdparty/MNN/include)
    message(STATUS "Found Target MNN from source code!")
  else()
    set(MNN_FOUND 0)
    message(STATUS "Not found Target MNN from source code!")
  endif()
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
