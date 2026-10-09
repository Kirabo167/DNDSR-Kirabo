# cmake/DndsApps.cmake
# Application executable lists and the ADD_EXE_APP helper function.
# Must be included after all library targets are defined.

# -------------------------------------------------------------------
# App executable lists
# -------------------------------------------------------------------

## app exes

set(DNDS_APPS_EXTERNAL
cgns_APITest
eigen_Test
STL_Test
json_Test
cgal_AABBTest
mpi_test
)

if(DNDS_USE_CANTERA)
  list(APPEND DNDS_APPS_EXTERNAL cantera_Test)
endif()

set(DNDS_APPS_EXTERNAL_CU
cuda_test
)

set(DNDS_APPS_DNDS
array_Test
arrayTrans_test
arrayDerived_test
arrayDOF_test
serializerJSON_Test
serializerH5_Test
stdPowerTest
objectPool_test
)

set(DNDS_APPS_DNDS_CU
array_cuda_Test
array_cuda_Bench
arrayDOF_test_cuda
)

set(DNDS_APPS_Geom
elements_Test
meshSerial_Test
ofReader_Test
partitionMeshSerial
)

set(DNDS_APPS_CFV
vrStatic_Test
vrBasic_Test
diffTensors_Test
)

set(DNDS_APPS_Solver
krylovTest
)



set(DNDS_APPS_Euler
gasTest
jacobiLUTest
oneDimProfileTest
)

# -------------------------------------------------------------------
# Symlinks for in-place development
# -------------------------------------------------------------------
if (NOT SKBUILD_PROJECT_NAME) # do not create links for SCIKIT-BUILD build
    install(CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/DNDSR/bin ${CMAKE_SOURCE_DIR}/src/bin SYMBOLIC)" COMPONENT py)
    install(CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/DNDSR/lib ${CMAKE_SOURCE_DIR}/src/lib SYMBOLIC)" COMPONENT py)
endif()
# Stub generation runs automatically as the final step of
# `cmake --install build --component py` (see DndsTooling.cmake).
# Generated .pyi files are install artifacts and remain ignored by git.

# -------------------------------------------------------------------
# ADD_EXE_APP helper function
# -------------------------------------------------------------------
set(EXE_SUFFIX "")
if (UNIX)
    set(EXE_SUFFIX ".exe")
endif()


function(ADD_EXE_APP EXES MAIN_DIR LIBS USE_EXCLUDE_FROM_ALL SRC_SUFFIX)
    message(STATUS "To add exes: ${EXES} with libs: ${LIBS}")
    foreach(EXE ${EXES})
        # add_executable(${EXE} app/external/${EXE}.cpp $<TARGET_OBJECTS:${OBJS}>)
        add_executable(${EXE} ${MAIN_DIR}/${EXE}.${SRC_SUFFIX})
        if(USE_EXCLUDE_FROM_ALL)
            set_target_properties(${EXE} PROPERTIES EXCLUDE_FROM_ALL ON)
        endif()
        target_link_libraries(${EXE} PUBLIC ${LIBS})
        target_link_libraries(${EXE} PUBLIC ${DNDS_EXTERNAL_LIBS})
        target_include_directories(${EXE} PUBLIC ${DNDS_EXTERNAL_INCLUDES} PUBLIC ${DNDS_INCLUDES})
        set_target_properties(${EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/app RUNTIME_OUTPUT_NAME "${EXE}${EXE_SUFFIX}")
        if(DNDS_RECORD_COMMIT)
            target_compile_definitions(${EXE} PUBLIC DNDS_CURRENT_COMMIT_HASH=${DNDS_RECORDED_COMMIT_HASH})
        endif()
        target_compile_definitions(${EXE} PUBLIC DNDS_VERSION_STRING="${DNDS_VERSION_FULL}")
        if( DNDS_USE_CCACHE )
            set_property(TARGET ${EXE} PROPERTY C_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
            set_property(TARGET ${EXE} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
            set_property(TARGET ${EXE} PROPERTY CUDA_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
        endif()
endforeach()

endfunction(ADD_EXE_APP)

# -------------------------------------------------------------------
# Register all app executables
# -------------------------------------------------------------------

# In topolocical order of the dependency graph
# euler_library
# euler_library_fast
# cfv
# geom
# dnds

## Mind That the TOPOLOGICAL ORDER should be obeyed!
ADD_EXE_APP("${DNDS_APPS_EXTERNAL}" "app/external" ";" ON cpp)
if(DNDS_USE_CANTERA AND DNDS_CANTERA_DATA_DIR)
    target_compile_definitions(cantera_Test PRIVATE CT_USE_SYSTEM_FMT=1)
    target_compile_definitions(cantera_Test PRIVATE DNDS_CANTERA_DATA_DIR="${DNDS_CANTERA_DATA_DIR}")
endif()
ADD_EXE_APP("${DNDS_APPS_DNDS}" "app/DNDS" "dnds;" ON cpp)
if(DNDS_USE_CUDA)
    ADD_EXE_APP("${DNDS_APPS_EXTERNAL_CU}" "app/external" ";" ON cu)
    ADD_EXE_APP("${DNDS_APPS_DNDS_CU}" "app/DNDS" "dnds;" ON cu)
endif()
ADD_EXE_APP("${DNDS_APPS_Solver}" "app/Solver" "dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_Geom}" "app/Geom" "geom;dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_CFV}" "app/CFV" "cfv;geom;dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_Euler}" "app/Euler" "cfv;geom;dnds;" ON cpp)

set(DNDS_APPS_Euler_Models)
foreach(item IN LISTS DNDS_Euler_Models_List)
    string(REPLACE "=" ";" keyval ${item})
    list(GET keyval 0 key)
    list(GET keyval 1 value)
    set(EXE_NAME "euler${value}")
    list(APPEND DNDS_APPS_Euler_Models ${EXE_NAME})
    ADD_EXE_APP("${EXE_NAME}" "app/Euler"
        "euler_library_${key};euler_library_fast_${key};cfv;geom;dnds;" ON cpp)
endforeach()

# Keep the upstream default executable; every other solver is opt-in.
set_target_properties(euler PROPERTIES EXCLUDE_FROM_ALL OFF)
foreach(dimension 2 3)
    ADD_EXE_APP("acm${dimension}D" "app/ACM"
        "acm_${dimension}D;cfv;geom;dnds;" ON cpp)
    ADD_EXE_APP("acmVariable${dimension}D" "app/ACMVariable"
        "acmVariable_${dimension}D;cfv;geom;dnds;" ON cpp)
    ADD_EXE_APP("ncfv_euler${dimension}D" "app/NCFV"
        "ncfv_${dimension}D;cfv;geom;dnds;" ON cpp)
endforeach()
ADD_EXE_APP("ACM" "app/ACM" "acm_3D;cfv;geom;dnds;" ON cpp)

ADD_EXE_APP("eulerState" "app/Euler" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds;" ON cpp)
set(DNDS_EULER_EXTRA_APPS eulerState)
if(DNDS_USE_CANTERA)
    ADD_EXE_APP("canteraConstVolTrajectory" "app/Euler" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds;" ON cpp)
    list(APPEND DNDS_EULER_EXTRA_APPS canteraConstVolTrajectory)
    target_compile_definitions(canteraConstVolTrajectory PRIVATE CT_USE_SYSTEM_FMT=1)
    if(DNDS_CANTERA_DATA_DIR)
        target_compile_definitions(canteraConstVolTrajectory PRIVATE DNDS_CANTERA_DATA_DIR="${DNDS_CANTERA_DATA_DIR}")
    endif()
endif()

# -------------------------------------------------------------------
# Aggregate convenience targets
# -------------------------------------------------------------------

add_custom_target(all_euler)
add_dependencies(all_euler ${DNDS_APPS_Euler_Models} ${DNDS_EULER_EXTRA_APPS})
add_custom_target(all_acm)
add_dependencies(all_acm ACM acm2D acm3D acmVariable2D acmVariable3D)
add_custom_target(all_ncfv_euler)
add_dependencies(all_ncfv_euler ncfv_euler2D ncfv_euler3D)
add_custom_target(all_solvers)
add_dependencies(all_solvers all_euler all_acm all_ncfv_euler)

if(DNDS_BUILD_TESTS AND Python_EXECUTABLE)
    add_test(NAME solver_case_configurations
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/check_solver_cases.py
            --build-dir ${CMAKE_BINARY_DIR})
    add_test(NAME solver_executable_selection
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/check_solver_cases.py
            --build-dir ${CMAKE_BINARY_DIR} --negative-only)
    set_tests_properties(solver_case_configurations solver_executable_selection PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        TIMEOUT 300 LABELS "solver_config" ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
