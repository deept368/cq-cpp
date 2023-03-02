cmake_minimum_required (VERSION 3.4.1)

project(bert-cpp)

set(CMAKE_CXX_STANDARD 14)
cmake_policy(SET CMP0074 OLD)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
option(PROFILE_ENABLE "Enable profile" ON)
file(GLOB DIR_SRCS src/bert/*.cc src/colbert/*.cc src/cq/*.cc)



    # add_definitions(-DPRFILE_FUNCTION)
    #  message("build profile binary")
     # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    #  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    #  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    #  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
    #  SET(CMAKE_CXX_FLAGS_RELEASE "-O0")
    #  SET(CMAKE_CXX_FLAGS_DEBUG "-O0")


include(FindProtobuf)
find_package(Protobuf REQUIRED)

set(MKL_ROOT /opt/intel/mkl)
include(FindMKL)
find_package(MKL REQUIRED)

find_package(Threads REQUIRED)



set(utf8proc_LIBRARY /usr/local/lib/libutf8proc.a)
set(utf8proc_INCLUDE_DIR /usr/local/include)

include_directories(${utf8proc_INCLUDE_DIR} ${MKL_INCLUDES})

add_library(bertcpp SHARED ${DIR_SRCS})

set(CMAKE_PREFIX_PATH ./libtorch)
find_package(Torch REQUIRED)
set(CMAKE_PREFIX_PATH "")
target_link_libraries(bertcpp ${TORCH_LIBRARIES} ${PROTOBUF_LIBRARY} ${MKL_LIBRARY} ${utf8proc_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

add_executable(bert_sample bert-sample.cpp)
target_link_libraries(bert_sample bertcpp)

# enable_testing()
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIR})

# file(GLOB TEST_SRCS bert/test/*.cpp)
# add_executable(bert_test ${TEST_SRCS})
# target_link_libraries(bert_test bertcpp ${GTEST_BOTH_LIBRARIES})
# add_test(unitest bert_test)
