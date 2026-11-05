# NCFV: Node Center Finite Volume Method.  It reuses DNDSR infrastructure
# without adding fields or algorithms to the existing solver modules.
set(LIBNAME ncfv_gpu)

set(CPPS
    NCFVConfig.cpp
    NCFVBoundary.cpp
    NCFVTopology.cpp
    NCFVDualGeometry.cpp
    NCFVNodeHalo.cpp
    NCFVReconstruction.cpp
)

set(LINKS
    dnds_gpu_flux
    dnds_gpu_resident
    cfv
    geom
    dnds
    ${DNDS_EXTERNAL_LIBS}
)

dnds_add_lib(${LIBNAME}_core "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)
foreach(dimension 2 3)
    dnds_add_lib(${LIBNAME}_${dimension}D "NCFVSpatial.cpp;NCFVSolver.cpp;NCFVResident.cpp"
        "${LIBNAME}_core;${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
    target_compile_definitions(${LIBNAME}_${dimension}D PRIVATE
        DNDS_NCFV_INSTANTIATION_DIM=${dimension})
endforeach()
add_library(${LIBNAME} INTERFACE)
target_link_libraries(${LIBNAME} INTERFACE ${LIBNAME}_2D ${LIBNAME}_3D)
