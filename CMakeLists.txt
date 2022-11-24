cmake_minimum_required( VERSION 3.14 )
project( simple_dvo )

set( CMAKE_CXX_STANDARD 20 )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
link_directories( ${PROJECT_SOURCE_DIR}/lib )

include_directories( ${PROJECT_SOURCE_DIR}/include )

add_subdirectory( ${PROJECT_SOURCE_DIR}/src )
