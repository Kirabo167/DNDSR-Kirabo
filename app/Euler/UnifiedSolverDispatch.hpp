/**
 * @file UnifiedSolverDispatch.hpp
 * @brief Lightweight declarations for model-specific unified-app launchers.
 */
#pragma once

namespace DNDS::App
{
    int RunEulerNS(int argc, char *argv[]);
    int RunEulerNS2D(int argc, char *argv[]);
    int RunEulerNS3D(int argc, char *argv[]);
    int RunEulerNSSA(int argc, char *argv[]);
    int RunEulerNSSA3D(int argc, char *argv[]);
    int RunEulerNS2EQ(int argc, char *argv[]);
    int RunEulerNS2EQ3D(int argc, char *argv[]);
    int RunEulerNSEX(int argc, char *argv[], int fieldNVariables);
    int RunEulerNSEX3D(int argc, char *argv[], int fieldNVariables);

    int RunACMConstantDensity2D(int argc, char *argv[]);
    int RunACMConstantDensity3D(int argc, char *argv[]);
    int RunACMVariableDensity2D(int argc, char *argv[]);
    int RunACMVariableDensity3D(int argc, char *argv[]);
    int RunNCFVIdealGas(int argc, char *argv[]);
}
