cmake_minimum_required (VERSION 2.6)
project (compiler)
# The version number.
set (COMPILER_VERSION_MAJOR 1)
set (COMPILER_VERSION_MINOR 0)


# LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
# query llvm libs
execute_process (
    COMMAND llvm-config --libs
    OUTPUT_VARIABLE llvm_libs
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# query llvm lib path
execute_process (
    COMMAND llvm-config --ldflags
    OUTPUT_VARIABLE llvm_ldflags
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
#message(STATUS "LLVM LIBS ${llvm_libs}")
# add llvm lib path to link flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${llvm_ldflags}")

# BISON
FIND_PACKAGE(BISON REQUIRED)
SET(BisonOutput ${CMAKE_SOURCE_DIR}/parser.cpp)
IF(BISON_FOUND)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${BisonOutput}
      COMMAND ${BISON_EXECUTABLE}
              --defines=${CMAKE_SOURCE_DIR}/tokens.h
              --output=${BisonOutput}
              ${CMAKE_SOURCE_DIR}/parser.y
      COMMENT "Generating parser.cpp"
    )
ENDIF()

# FLEX
FIND_PACKAGE(FLEX REQUIRED)
SET(FlexOutput ${CMAKE_SOURCE_DIR}/tokens.cpp)
IF(FLEX_FOUND)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${FlexOutput}
      COMMAND ${FLEX_EXECUTABLE}
              --outfile=${FlexOutput}
              ${CMAKE_SOURCE_DIR}/tokens.l
      COMMENT "Generating tokens.cpp"
    )
ENDIF()


# nGraph
set(NGRAPH_INCLUDE_PATH "/home/louis/git/aipg/install/ngraph-debug/include")
set(NGRAPH_LIB_PATH "/home/louis/git/aipg/install/ngraph-debug/lib")

include_directories(${NGRAPH_INCLUDE_PATH})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${NGRAPH_LIB_PATH}")
# set rpath for linking
SET(CMAKE_BUILD_RPATH ${NGRAPH_LIB_PATH})
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# SOURCE CODE
set(SRC
    codegen.cpp
    corefn.cpp
    main.cpp
    native.cpp
    parser.cpp
    test_tools.cpp
    ${BisonOutput}
    ${FlexOutput})

# add the executable
add_executable(compiler ${SRC})
target_link_libraries(compiler ${llvm_libs} ngraph pthread dl ncurses)
