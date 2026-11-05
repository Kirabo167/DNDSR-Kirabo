#pragma once
#include "ResidentNCFV.hpp"
#include "FluxBatch.hpp"

namespace DNDS::GPU
{
    inline constexpr int ACMMaximumModes = 19;
    struct ResidentACMCell
    {
        double volume = 0, smoothScale = 1, relaxation = 1, wallDistance = 1;
        int modes = 0;
        DeviceIndex begin = 0, end = 0;
    };
    struct ResidentACMBoundarySample
    {
        double normal[3]{}, prescribed[4]{}, basis[ACMMaximumModes]{}, response[ACMMaximumModes]{};
        int type = 0;
    };
    struct ResidentACMLink
    {
        DeviceIndex neighbor = -1, face = 0, coupling = 0, boundaryBegin = 0, boundaryEnd = 0;
        int side = 0;
        double rotation[9]{}, meanWeight[ACMMaximumModes]{}, gradientWeight[3]{}, turbulenceWeight[3]{};
    };
    struct ResidentACMPoint
    {
        double normal[3]{}, measure = 0, displacement[3]{};
        double delta[2][3]{}, basis[2][ACMMaximumModes]{}, gradient[2][ACMMaximumModes * 3]{};
    };
    struct ResidentACMFace
    {
        DeviceIndex left = 0, right = -1, begin = 0, end = 0;
        double normal[3]{}, prescribed[4]{}, rotation[9]{}, area = 0, wallDistance = 1;
        int type = 0;
    };
    struct ResidentACMSettings
    {
        FluxInput flow;
        int dimension = 3, reconstruction = 2, limiterType = 0, integrator = 0;
        int sweeps = 3, maximumSweeps = 10000, checkInterval = 10;
        bool resetCoefficients = false, sor = true, limiter = true, controlled = false, reconstructionGMRES = false;
        double reconstructionTolerance = 0, reconstructionRelaxation = .7;
        bool viscous = false, cflStep = false, localStep = true;
        double viscosity = 0, cfl = .5, fixedStep = .001, maximumStep = 1e100;
        int implicitIterations = 20, lusgsSweeps = 2, gmresSubspace = 10, gmresRestarts = 3;
        double implicitTolerance = 1e-8, implicitRelaxation = 1, linearRelativeTolerance = 1e-6, linearAbsoluteTolerance = 1e-10;
        double physicalStep = .01;
        bool sst = false, turbulenceSecondOrder = true, turbulenceSource = true, requireConvergence = true;
        double turbulenceMinimum[2]{1e-12, 1e-10}, turbulenceMaximum[2]{1e6, 1e12}, turbulenceFar[2]{1e-6, 1};
        double eddyRatioMaximum = 1e5, wallOmegaCoefficient = 800, minimumWallDistance = 1e-10, turbulenceTimeScale = 1;
        int convergenceCriterion = 0;
        double flowRelativeTolerance = 0, turbulenceRelativeTolerance = 0;
        double residualScales[6]{1, 1, 1, 1, 1, 1}, relativeTolerances[6]{};
    };
    struct ResidentACMData
    {
        ResidentACMSettings settings;
        DeviceIndex owned = 0;
        std::vector<DeviceIndex> globals, reconstructionNodes, reconstructionLevels, backwardNodes, backwardLevels;
        std::vector<ResidentACMCell> cells;
        std::vector<ResidentACMLink> links;
        std::vector<ResidentACMBoundarySample> boundaries;
        std::vector<ResidentACMFace> faces;
        std::vector<ResidentACMPoint> points;
        std::vector<double> coupling, initialState, initialTurbulence;
    };
    struct ResidentACMReport
    {
        double initialNorm = 0, finalNorm = 0, minimumStep = 0;
        int iterations = 0;
        bool converged = false;
        double initialTurbulenceNorm = 0, turbulenceNorm = 0;
        std::array<double, 6> initialEquationNorms{}, equationNorms{};
    };
    class ResidentACM
    {
        struct Impl;
        std::unique_ptr<Impl> _impl;

    public:
        ResidentACM(const ResidentACMData &data, MPI_Comm communicator);
        ~ResidentACM();
        ResidentACMReport Advance();
        void Download(std::vector<double> &state);
        void DownloadTurbulence(std::vector<double> &state);
        void Report(int rank) const;
    };
}
