# @file CMakeLists.txt
# @brief Build definition for the standalone constant-density ACM library.
# @note Modifier: Runzhi Ma.

set(LIBNAME acm)

set(CPPS
    ACM.cpp
    ACMBDF2.cpp
    ACMTime.cpp
    ACMTurbulence.cpp
    acm2D.cpp
    acm3D.cpp
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

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)
target_sources(${LIBNAME} PRIVATE ${HPPS})
