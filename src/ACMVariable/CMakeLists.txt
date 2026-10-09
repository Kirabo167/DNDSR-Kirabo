# Independent variable-density ACM assembly. Author: Runzhi Ma.
set(LIBNAME acmVariable)
set(CPPS ACM.cpp ACMBDF2.cpp ACMTime.cpp ACMTurbulence.cpp)
set(LINKS cfv geom dnds ${DNDS_EXTERNAL_LIBS})
dnds_add_lib(${LIBNAME}_core "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
foreach(dimension 2 3)
    dnds_add_lib(${LIBNAME}_${dimension}D "acm${dimension}D.cpp"
        "${LIBNAME}_core;${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
endforeach()
add_library(${LIBNAME} INTERFACE)
target_link_libraries(${LIBNAME} INTERFACE ${LIBNAME}_2D ${LIBNAME}_3D)
