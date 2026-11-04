cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TestParticlesPaperHIP LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/)
#find_package(hip)

include_directories(/opt/rocm/include)
add_compile_definitions( __HIP_PLATFORM_AMD__ )
enable_language(HIP)

set(GPU_TARGET "gfx90a" CACHE STRING "GPU target to compile for")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=${GPU_TARGET}")

# Find all code files
file(	
    GLOB_RECURSE
    source_tests_files
    ./*.hip
    )

foreach(exec ${source_tests_files}) 
    get_filename_component(
	    execname ${exec}
	    NAME_WE
    )

    if($ENV{VERBOSE})
        MESSAGE(STATUS "-- Add ${execname}")
    endif()

    # set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
    add_executable(
		    ${execname}
		    ${exec}
	    )

    # target_link_libraries(${execname}  hip::device hip::host)
endforeach(exec)

