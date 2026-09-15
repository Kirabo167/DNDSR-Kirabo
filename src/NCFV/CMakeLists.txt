# NCFV: Node Center Finite Volume Method.  It reuses DNDSR infrastructure
# without adding fields or algorithms to the existing solver modules.
set(LIBNAME ncfv)

set(CPPS
    NCFVConfig.cpp
    NCFVBoundary.cpp
    NCFVTopology.cpp
    NCFVDualGeometry.cpp
    NCFVNodeHalo.cpp
    NCFVReconstruction.cpp
    NCFVSpatial.cpp
    NCFVSolver.cpp
)

set(LINKS
    cfv
    geom
    dnds
    ${DNDS_EXTERNAL_LIBS}
)

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)

add_executable(NCFV ${PROJECT_SOURCE_DIR}/app/NCFV/NCFV.cpp)
set_target_properties(NCFV PROPERTIES
    EXCLUDE_FROM_ALL ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app
    RUNTIME_OUTPUT_NAME "NCFV.exe")
target_include_directories(NCFV PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(NCFV PRIVATE ncfv cfv geom dnds ${DNDS_EXTERNAL_LIBS})
target_compile_definitions(NCFV PRIVATE
    DNDS_VERSION_STRING="${DNDS_VERSION_FULL}")
if(DNDS_USE_CCACHE)
    set_property(TARGET NCFV PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
endif()
