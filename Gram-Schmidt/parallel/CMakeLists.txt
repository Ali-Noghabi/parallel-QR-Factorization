cmake_minimum_required(VERSION 3.10)
project(GramSchmidtParallel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the MPI paths manually
set(MPI_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
set(MPI_C_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

include_directories(${MPI_INCLUDE_PATH})

# Add the executable
add_executable(GramSchmidtParallel GramSchmidtParallel.cpp)

# Link against MPI libraries
target_link_libraries(GramSchmidtParallel ${MPI_CXX_LIBRARIES})

# If you want to include MPI executable paths
set(MPIEXEC "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")

# Information message to verify the paths
message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI CXX libraries: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable path: ${MPIEXEC}")
