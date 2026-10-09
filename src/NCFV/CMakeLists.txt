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
)

set(LINKS
    cfv
    geom
    dnds
    ${DNDS_EXTERNAL_LIBS}
)

dnds_add_lib(${LIBNAME}_core "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)
foreach(dimension 2 3)
    dnds_add_lib(${LIBNAME}_${dimension}D "NCFVSpatial.cpp;NCFVSolver.cpp"
        "${LIBNAME}_core;${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
    target_compile_definitions(${LIBNAME}_${dimension}D PRIVATE
        DNDS_NCFV_INSTANTIATION_DIM=${dimension})
endforeach()
add_library(${LIBNAME} INTERFACE)
target_link_libraries(${LIBNAME} INTERFACE ${LIBNAME}_2D ${LIBNAME}_3D)

# Accuracy/resource probe used by the 3-D NCFV diagnostic cases.
add_executable(ncfv_reconstruction_benchmark
    ${CMAKE_SOURCE_DIR}/cases/ncfv_euler/3D/diagnostics/benchmark_reconstruction.cpp)
target_include_directories(ncfv_reconstruction_benchmark PRIVATE
    ${DNDS_EXTERNAL_INCLUDES}
    ${DNDS_INCLUDES})
target_link_libraries(ncfv_reconstruction_benchmark PRIVATE
    ncfv cfv geom dnds ${DNDS_EXTERNAL_LIBS})
set_target_properties(ncfv_reconstruction_benchmark PROPERTIES
    EXCLUDE_FROM_ALL ON)

# Instantaneous momentum-RHS accuracy against the exact 2-D incompressible
# Taylor--Green field, embedded in a fully periodic 3-D Cartesian mesh.
add_executable(ncfv_taylor_green_2d_rhs_probe
    ${CMAKE_SOURCE_DIR}/cases/ncfv_euler/3D/diagnostics/taylor_green_2d_rhs_probe.cpp)
target_include_directories(ncfv_taylor_green_2d_rhs_probe PRIVATE
    ${DNDS_EXTERNAL_INCLUDES}
    ${DNDS_INCLUDES})
target_link_libraries(ncfv_taylor_green_2d_rhs_probe PRIVATE
    ncfv cfv geom dnds ${DNDS_EXTERNAL_LIBS})
set_target_properties(ncfv_taylor_green_2d_rhs_probe PROPERTIES
    EXCLUDE_FROM_ALL ON)
