cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.10)
include(ExternalProject)
include(ProcessorCount)
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
SET(AVX ON)
add_definitions(-DNAPI_VERSION=6)
ProcessorCount(N)
if(NOT N EQUAL 0)
  set(BUILD_FLAGS -j ${N})
endif()
set (CMAKE_CXX_STANDARD 11)
set (PADDLE_PATH ${CMAKE_CURRENT_BINARY_DIR}/paddle)

project (node-paddleocr)

include_directories(${CMAKE_JS_INC})

# Include Node-API wrappers
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        )
string(REGEX REPLACE "[\r\n\"]" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})


# define NPI_VERSION
add_definitions(-DNAPI_EXPERIMENTAL)
add_definitions("/DPD_INFER_DECL=")

SET(PADDLE_BUILD_PATH ${PADDLE_PATH}/src/paddle/build)
cmake_path(NATIVE_PATH PADDLE_BUILD_PATH PADDLE_BUILD_PATH_CMD)

ExternalProject_Add(
    paddle
    GIT_REPOSITORY https://github.com/PaddlePaddle/Paddle.git
    GIT_TAG release/2.1
    GIT_SHALLOW 1
    PREFIX ${PADDLE_PATH}
    TMP_DIR        ${PADDLE_PATH}/tmp
    STAMP_DIR      ${PADDLE_PATH}/src/paddle-stamp
    DOWNLOAD_DIR   ${PADDLE_PATH}/src
    SOURCE_DIR     ${PADDLE_PATH}/src/paddle
    BINARY_DIR     ${PADDLE_PATH}/src/paddle/build
    LOG_DIR        ${PADDLE_PATH}/src/paddle-stamp
    CMAKE_ARGS -T host=x64 -Wno-dev -DCMAKE_CONFIGURATION_TYPES=${CMAKE_BUILD_TYPE} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DWITH_MKL=ON -DWITH_MKLDNN=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DWITH_TENSORRT=OFF -DWITH_AVX=ON -DMSVC_STATIC_CRT=OFF -DWITH_STATIC_LIB=ON -DWITH_AVX=${AVX}
    BUILD_COMMAND cmake --build . --target inference_lib_dist ${BUILD_FLAGS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} --config ${CMAKE_BUILD_TYPE} -A x64 -T host=x64
    PATCH_COMMAND if not exist "${PADDLE_BUILD_PATH_CMD}" mkdir "${PADDLE_BUILD_PATH_CMD}"
    COMMAND echo git apply ${CMAKE_SOURCE_DIR}/paddle_cmake.patch
    COMMAND git apply ${CMAKE_SOURCE_DIR}/paddle_cmake.patch
    INSTALL_COMMAND ""
)

SET(PADDLE_LIB ${PADDLE_PATH}/src/paddle/build/paddle_inference_install_dir)


file(GLOB SOURCE_FILES
    paddleocr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")

include_directories("${PADDLE_LIB}/paddle/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2")

set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
include_directories("${MATH_LIB_PATH}/include")
set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
endif()
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/mklml.dll ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
    COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
)
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} ${MATH_LIB} ${MKLDNN_LIB} ${EXTERNAL_LIB} shlwapi.lib ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(${PROJECT_NAME} "${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib/glog.lib")
target_link_libraries(${PROJECT_NAME} "${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib/gflags_static.lib")
target_link_libraries(${PROJECT_NAME} "${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib/libprotobuf.lib")
target_link_libraries(${PROJECT_NAME} "${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib/xxhash.lib")
target_link_libraries(${PROJECT_NAME} "${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/lib/cryptopp-static.lib")
target_link_libraries(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/lib/opencv_core452.lib")
target_link_libraries(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/lib/opencv_imgproc452.lib")
add_dependencies(${PROJECT_NAME} paddle)