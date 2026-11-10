# @file CMakeLists.txt
# @brief Build definition for the standalone constant-density ACM library.
# @note Modifier: Runzhi Ma.

set(LIBNAME acm_gpu)

set(CPPS
    ACM.cpp
    ACMBDF2.cpp
    ACMTime.cpp
    ACMTurbulence.cpp
)

set(HPPS
    ACM.hpp
    ACMBC.hpp
    ACMBDF2.hpp
    ACMConfig.hpp
    ACMEvaluator.hpp
    ACMEvaluator.hxx
    ACMFlux.hpp
    ACMParallel.hpp
    ACMSettings.hpp
    ACMSolver.hpp
    ACMSolver.hxx
    ACMModelState.hpp
    ACMRANSSolver.hpp
    ACMRANSSolver.hxx
    ACMState.hpp
    ACMTime.hpp
    ACMTurbulence.hpp
    ACMTurbulenceTransport.hpp
    ACMTurbulenceTransport.hxx
    SingleBlockApp.hpp
)

set(LINKS
    dnds_gpu_resident
    dnds_gpu_flux
    cfv
    geom
    dnds
    ${DNDS_EXTERNAL_LIBS}
)

dnds_add_lib(${LIBNAME}_core "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)
target_sources(${LIBNAME}_core PRIVATE ${HPPS})
# dnds_add_lib uses legacy link-interface wiring, so the resident library's
# compile definitions do not propagate into these object compilations.
# Keep every producer/consumer of ResidentACMData on the same struct layout.
target_compile_definitions(${LIBNAME}_core PUBLIC DNDS_GPU_ACM_MAXIMUM_MODES=${DNDS_GPU_ACM_MAXIMUM_MODES})
foreach(dimension 2 3)
    dnds_add_lib(${LIBNAME}_${dimension}D "acm${dimension}D.cpp"
        "${LIBNAME}_core;${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
    dnds_add_lib(${LIBNAME}_rans_${dimension}D "acmRANS${dimension}D.cpp"
        "${LIBNAME}_${dimension}D;${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
    target_compile_definitions(${LIBNAME}_${dimension}D PUBLIC DNDS_GPU_ACM_MAXIMUM_MODES=${DNDS_GPU_ACM_MAXIMUM_MODES})
    target_compile_definitions(${LIBNAME}_rans_${dimension}D PUBLIC DNDS_GPU_ACM_MAXIMUM_MODES=${DNDS_GPU_ACM_MAXIMUM_MODES})
endforeach()
# Existing module tests can still exercise both dimensions through this target.
add_library(${LIBNAME} INTERFACE)
target_link_libraries(${LIBNAME} INTERFACE ${LIBNAME}_rans_2D ${LIBNAME}_rans_3D)
