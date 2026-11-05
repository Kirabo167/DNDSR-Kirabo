#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <mpi.h>

namespace DNDS::GPU
{
    using DeviceIndex = std::int64_t;

    struct ResidentNode
    {
        double volume = 0, reference[3]{1, 1, 1}, coordinate[3]{};
        double basisMean[9]{};
        DeviceIndex recBegin = 0, recEnd = 0, recoveryBegin = 0, recoveryEnd = 0;
        DeviceIndex edgeBegin = 0, edgeEnd = 0, boundaryBegin = 0, boundaryEnd = 0;
        DeviceIndex variationalBegin = 0, variationalEnd = 0;
    };
    struct ResidentRecEntry
    {
        DeviceIndex node = 0;
        double weights[9]{};
    };
    struct ResidentVariationalEntry
    {
        DeviceIndex node = 0;
        double coupling[81]{}, jump[9]{};
    };
    struct ResidentRecoveryEntry
    {
        DeviceIndex node = 0;
        double weight[3]{};
    };
    struct ResidentSharedTrace
    {
        DeviceIndex anchor = 0, begin = 0, end = 0;
    };
    struct ResidentSharedFace
    {
        DeviceIndex traces[2]{}, begin = 0, end = 0;
    };
    struct ResidentSurfaceEntry
    {
        DeviceIndex node = 0;
        double stateWeight[2][3]{}, fluxWeight[2][9]{}, gradientWeight = 0;
    };
    struct ResidentQuadrature
    {
        double coordinate[3]{}, normal[3]{}, weight = 0;
    };
    struct ResidentEdge
    {
        DeviceIndex nodes[2]{}, begin = 0, end = 0, quadBegin = 0, quadEnd = 0;
        double vectorMeasure[3]{}, measure = 0;
    };
    struct ResidentIncidence
    {
        DeviceIndex edge = 0, neighbor = 0;
        int side = 0, sign = 1;
    };
    struct ResidentBoundaryEntry
    {
        DeviceIndex node = 0;
        double weight = 0;
    };
    struct ResidentBoundary
    {
        DeviceIndex anchor = 0, begin = 0, end = 0, quadBegin = 0, quadEnd = 0;
        double normal[3]{}, measure = 0, prescribed[5]{}, wallVelocity[3]{};
        double pressure = 1, wallTemperature = 1;
        int mode = 0, zone = 0;
        bool strong = false;
    };
    struct ResidentNCFVSettings
    {
        int dimension = 3;
        bool traditional = false, limiter = true, limitRecovery = false, binaryBarth = false;
        bool limitPressure = false, directionalBounds = false;
        bool fluxCorrection = false, binaryFluxCorrection = false, entropyFluxCorrection = false;
        bool pairFluxBounds = false;
        double maximumFluxCorrection = 1;
        bool viscous = false;
        int viscosityModel = 0;
        double viscosity = 0, prandtl = .72, gasConstant = 1;
        double referenceTemperature = 1, sutherlandConstant = 0, viscousSpectralFactor = 4;
        bool cflStep = false, localStep = true, rejectNonphysical = false;
        double gamma = 1.4, fixedStep = 1e-4, cfl = .5, minimumStep = 1e-12, maximumStep = 1e10;
        double periodic[3]{};
        int maximumRetries = 12;
        bool variational = false;
        int variationalIterations = 30;
        double variationalRelaxation = 1;
        bool sharedGradient = false, sharedParallel = false, sharedBudget = false;
    };
    struct ResidentNCFVData
    {
        ResidentNCFVSettings settings;
        DeviceIndex ownedNodes = 0, ownedEdges = 0;
        std::vector<DeviceIndex> nodeGlobals, edgeGlobals;
        std::vector<ResidentNode> nodes;
        std::vector<ResidentRecEntry> reconstruction;
        std::vector<ResidentVariationalEntry> variational;
        std::vector<DeviceIndex> variationalNodes, variationalLevels;
        std::vector<DeviceIndex> sharedGlobals, sharedSupport, sharedFacesByGroup, sharedGroups;
        std::vector<ResidentSharedTrace> sharedTraces;
        std::vector<ResidentSharedFace> sharedFaces;
        std::vector<ResidentRecoveryEntry> sharedWeights;
        std::vector<ResidentRecoveryEntry> recovery;
        std::vector<ResidentSurfaceEntry> surface;
        std::vector<ResidentEdge> edges;
        std::vector<ResidentIncidence> incidence;
        std::vector<ResidentBoundary> boundaries;
        std::vector<ResidentBoundaryEntry> boundarySupport;
        std::vector<ResidentQuadrature> quadrature;
        std::vector<double> initialState;
    };
    struct ResidentStepReport
    {
        double residual = 0, minimumStep = 0, maximumStep = 0;
        int retries = 0;
    };

    class ResidentNCFV
    {
        struct Impl;
        std::unique_ptr<Impl> _impl;

    public:
        ResidentNCFV(const ResidentNCFVData &data, MPI_Comm communicator);
        ~ResidentNCFV();
        ResidentNCFV(const ResidentNCFV &) = delete;
        ResidentNCFV &operator=(const ResidentNCFV &) = delete;
        ResidentStepReport Advance(double maximumStep);
        double EvaluateResidual();
        void Download(std::vector<double> &state, std::vector<double> *steps = nullptr);
        void Report(int rank) const;
    };
}
