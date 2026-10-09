# @file CMakeLists.txt
# @brief Build definition for the standalone constant-density ACM library.
# @note Modifier: Runzhi Ma.

set(LIBNAME acm)

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
    ACMState.hpp
    ACMTime.hpp
    ACMTurbulence.hpp
    ACMTurbulenceTransport.hpp
    ACMTurbulenceTransport.hxx
    SingleBlockApp.hpp
)

set(LINKS
    cfv
    geom
    dnds
    ${DNDS_EXTERNAL_LIBS}
)

dnds_add_lib(${LIBNAME}_core "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)
target_sources(${LIBNAME}_core PRIVATE ${HPPS})
foreach(dimension 2 3)
    dnds_add_lib(${LIBNAME}_${dimension}D "acm${dimension}D.cpp"
        "${LIBNAME}_core;${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
endforeach()
# Existing module tests can still exercise both dimensions through this target.
add_library(${LIBNAME} INTERFACE)
target_link_libraries(${LIBNAME} INTERFACE ${LIBNAME}_2D ${LIBNAME}_3D)
