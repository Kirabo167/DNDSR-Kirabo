/** Isolated static, periodic HOCFV implementation from the supplied complete specification. */
#include "DNDS/Errors.hpp"
#include "Geom/CGNS.hpp"
#include "Geom/Quadrature.hpp"
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace
{
    using V = Eigen::Vector3d;
    using U = Eigen::Matrix<double, 5, 1>;
    using B = Eigen::Matrix<double, 9, 1>;
    using C = Eigen::Matrix<double, 9, 5>;
    using A = Eigen::Matrix<double, 9, 9>;
    using Rows = Eigen::Matrix<double, 10, 9>;
    constexpr double gammaGas = 1.4;
    constexpr double pi = 3.1415926535897932384626433832795;
    V domain(10, 10, 4);
    const std::array<std::array<int, 3>, 9> powers{{
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {1, 1, 0},
        {0, 2, 0}, {0, 1, 1}, {0, 0, 2}, {1, 0, 1}}};
    const std::array<std::array<int, 3>, 10> derivatives{{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0},
        {1, 1, 0}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}, {1, 0, 1}}};

    template <class T>
    nlohmann::json Array(const T &matrix)
    {
        nlohmann::json result = nlohmann::json::array();
        for (int i = 0; i < matrix.rows(); i++)
        {
            nlohmann::json row = nlohmann::json::array();
            for (int j = 0; j < matrix.cols(); j++)
                row.push_back(matrix(i, j));
            result.push_back(row);
        }
        return result;
    }

    void Cgns(int error)
    {
        DNDS_check_throw_info(error == CG_OK, cg_get_error());
    }

    double Pressure(const U &u)
    {
        return (gammaGas - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
    }

    bool Physical(const U &u)
    {
        return u.allFinite() && u(0) > 0 && u(4) > 0 && Pressure(u) > 0;
    }

    U Pack(double rho, const V &velocity, double pressure)
    {
        U result;
        result << rho, rho * velocity,
            pressure / (gammaGas - 1) + 0.5 * rho * velocity.squaredNorm();
        return result;
    }

    U PhysicalFlux(const U &u, const V &n)
    {
        const V velocity = u.segment<3>(1) / u(0);
        const double p = Pressure(u), mass = u(0) * velocity.dot(n);
        U result;
        result << mass, mass * velocity + p * n,
            (u(4) + p) * velocity.dot(n);
        return result;
    }

    U Roe(const U &left, const U &right, const V &n)
    {
        DNDS_check_throw_info(Physical(left) && Physical(right), "Reference Roe received illegal state");
        const V vl = left.segment<3>(1) / left(0), vr = right.segment<3>(1) / right(0);
        const double pl = Pressure(left), pr = Pressure(right);
        const double ratio = std::sqrt(right(0) / left(0));
        const double rho = std::sqrt(left(0) * right(0));
        const V velocity = (vl + ratio * vr) / (1 + ratio);
        const double h = ((left(4) + pl) / left(0) + ratio * (right(4) + pr) / right(0)) / (1 + ratio);
        const double a2 = (gammaGas - 1) * (h - 0.5 * velocity.squaredNorm());
        DNDS_check_throw_info(a2 > 0, "Reference Roe has invalid averaged sound speed");
        const double sound = std::sqrt(a2), un = velocity.dot(n);
        const auto fixed = [&](double value)
        {
            const double lambda = std::abs(value), delta = 0.15 * sound;
            return lambda < delta ? 0.5 * (lambda * lambda / delta + delta) : lambda;
        };
        const V dv = vr - vl;
        const double dp = pr - pl, dun = dv.dot(n);
        const double b0 = fixed(un) * (right(0) - left(0) - dp / a2);
        const double bp = 0.5 * fixed(un + sound) * (dp + rho * sound * dun) / a2;
        const double bm = 0.5 * fixed(un - sound) * (dp - rho * sound * dun) / a2;
        const double mass = b0 + bp + bm, normal = sound * (bp - bm);
        const V tangent = fixed(un) * rho * (dv - n * dun);
        U dissipation;
        dissipation << mass, velocity * mass + n * normal + tangent,
            h * mass + un * normal + velocity.dot(tangent) - a2 * b0 / (gammaGas - 1);
        return 0.5 * (PhysicalFlux(left, n) + PhysicalFlux(right, n) - dissipation);
    }

    B Basis(const V &point, const V &center, const V &ref, const B &moment, int derivative = 0)
    {
        const V xi = (point - center).cwiseQuotient(ref);
        B result;
        for (int j = 0; j < 9; j++)
        {
            double value = 1;
            for (int d = 0; d < 3; d++)
            {
                const int a = powers[j][d], order = derivatives[derivative][d];
                if (order > a)
                {
                    value = 0;
                    break;
                }
                for (int k = 0; k < order; k++)
                    value *= (a - k) / ref(d);
                for (int k = 0; k < a - order; k++)
                    value *= xi(d);
            }
            result(j) = value - (derivative == 0 ? moment(j) : 0);
        }
        return result;
    }

    Rows DifferentialRows(const V &point, const V &center, const V &ref, const B &moment, double length)
    {
        Rows rows;
        for (int r = 0; r < 10; r++)
        {
            double weight = r == 0 ? 1 : r < 4 ? 0.5 * length : 0.25 * length * length;
            if (r == 5 || r == 7 || r == 9)
                weight *= std::sqrt(2.0);
            rows.row(r) = weight * Basis(point, center, ref, moment, r).transpose();
        }
        return rows;
    }

    struct Tet
    {
        std::array<V, 4> points;
        double volume = 0;
    };
    struct Face
    {
        std::vector<int> nodes;
        std::vector<int> cells;
        V center = V::Zero();
    };
    struct Edge
    {
        int left = 0, right = 0;
        V midpoint = V::Zero(), areaVector = V::Zero();
    };
    struct Sample
    {
        int neighbor = 0;
        V point = V::Zero(), otherPoint = V::Zero();
        double length = 0;
    };
    struct Patch
    {
        int owner = 0, partner = 0;
        V point = V::Zero(), otherPoint = V::Zero(), areaVector = V::Zero();
        std::array<std::array<V, 3>, 2> triangles;
    };
    struct SurfacePiece
    {
        std::array<V, 3> points;
        V areaVector = V::Zero();
    };
    struct QuadratureFace
    {
        int left = 0, right = 0;
        bool internal = true;
        V normal = V::Zero();
        B leftBasis = B::Zero(), rightBasis = B::Zero();
        double weight = 0;
    };

    std::array<std::pair<Eigen::Vector2d, double>, 6> TriangleRule()
    {
        const double a = 0.091576213509771, b = 0.8168475729804585;
        const double c = 0.445948490915965, d = 0.108103018168070;
        return {{{Eigen::Vector2d(a, a), 0.109951743655322}, {Eigen::Vector2d(b, a), 0.109951743655322},
                 {Eigen::Vector2d(a, b), 0.109951743655322}, {Eigen::Vector2d(c, c), 0.2233815896780115},
                 {Eigen::Vector2d(d, c), 0.2233815896780115}, {Eigen::Vector2d(c, d), 0.2233815896780115}}};
    }
    struct Coupling
    {
        int neighbor = 0;
        A matrix = A::Zero();
        B jump = B::Zero();
    };

    std::array<std::pair<V, double>, 8> TetRule()
    {
        const double a = 0.106994014770537, b = 0.6790179556883893;
        const double c = 0.3280585716625127, d = 0.0158242850124619;
        const double w1 = 0.75 * 0.1487451997767295, w2 = 0.75 * 0.1845881335566038;
        return {{{V(a, a, b), w1}, {V(a, b, a), w1}, {V(b, a, a), w1}, {V(a, a, a), w1},
                 {V(c, c, d), w2}, {V(c, d, c), w2}, {V(d, c, c), w2}, {V(c, c, c), w2}}};
    }

    class Reference
    {
        int _rank = 0, _ranks = 1, _first = 0, _last = 0;
        std::vector<int> _counts5, _offsets5, _counts45, _offsets45;
        std::vector<int> _haloSendNodes, _haloRecvNodes;
        std::array<std::vector<int>, 2> _haloSendCounts, _haloRecvCounts, _haloSendOffsets, _haloRecvOffsets;
        std::array<std::vector<double>, 2> _haloSendBuffer, _haloRecvBuffer;
        std::vector<V> _x, _ref;
        std::vector<B> _moments;
        std::vector<double> _volume, _tetVolume;
        std::vector<std::array<int, 6>> _cells;
        std::vector<std::array<int, 5>> _cellFaces;
        std::vector<V> _cellCenters;
        std::vector<Face> _faces;
        std::vector<Edge> _edges;
        std::vector<Patch> _patches;
        std::vector<std::vector<SurfacePiece>> _surfacePieces;
        std::vector<QuadratureFace> _quadratureFaces;
        V _firstQuadraturePoint = V::Zero();
        bool _keepSurfacePieces = false, _quadratureFlux = false;
        bool _analyticFaces = false, _analyticTaylor = false, _skipReconstruction = false;
        bool _entropyMomentCorrection = false;
        std::vector<int> _edgeMomentIndex;
        std::vector<std::array<B, 2>> _edgeMomentOperators, _patchMomentOperators;
        std::string _problem = "entropy";
        std::vector<std::vector<std::pair<int, double>>> _incidence;
        std::vector<std::vector<Tet>> _tets;
        std::vector<std::vector<Sample>> _samples;
        std::vector<std::vector<Coupling>> _operators;
        std::vector<A> _matrices;
        std::vector<C> _coefficients;
        std::vector<U> _state;
        long long _fallbackCount = 0, _rhsCalls = 0, _faceCalls = 0;
        double _maximumVolumeMismatch = 0, _maximumClosure = 0, _maximumMomentMismatch = 0;
        double _maximumCondition = 0, _totalVolume = 0, _maximumMatrixSolveError = 0;
        double _firstReconstructionResidual = 0, _lastReconstructionResidual = 0;
        nlohmann::json _audit;
        bool Owned(int i) const { return _first <= i && i < _last; }

        B TriangleMeanBasis(const std::array<V, 3> &triangle, int node) const
        {
            V sum = V::Zero();
            Eigen::Matrix3d squareSum = Eigen::Matrix3d::Zero();
            for (const V &point : triangle)
            {
                const V xi = (point - _x[node]).cwiseQuotient(_ref[node]);
                sum += xi; squareSum.noalias() += xi * xi.transpose();
            }
            const V first = sum / 3;
            const Eigen::Matrix3d second = (sum * sum.transpose() + squareSum) / 12;
            B result;
            result << first, second(0, 0), second(0, 1), second(1, 1),
                second(1, 2), second(2, 2), second(0, 2);
            return result - _moments[node];
        }

        void BuildEntropyMomentOperators()
        {
            if (!_keepSurfacePieces) return;
            const V velocity(1, 1, 0);
            _edgeMomentIndex.assign(_edges.size(), -1);
            for (std::size_t id = 0; id < _edges.size(); id++)
            {
                const auto &edge = _edges[id];
                if (!Owned(edge.left) && !Owned(edge.right)) continue;
                std::array<B, 2> integral{B::Zero(), B::Zero()};
                for (const auto &piece : _surfacePieces[id])
                {
                    const double advectiveArea = piece.areaVector.dot(velocity);
                    integral[0] += advectiveArea * TriangleMeanBasis(piece.points, edge.left);
                    integral[1] += advectiveArea * TriangleMeanBasis(piece.points, edge.right);
                }
                const double area = edge.areaVector.dot(velocity);
                integral[0] -= area * Basis(edge.midpoint, _x[edge.left], _ref[edge.left], _moments[edge.left]);
                integral[1] -= area * Basis(edge.midpoint, _x[edge.right], _ref[edge.right], _moments[edge.right]);
                _edgeMomentIndex[id] = static_cast<int>(_edgeMomentOperators.size());
                _edgeMomentOperators.push_back(integral);
            }
            for (const auto &patch : _patches)
            {
                std::array<B, 2> integral{B::Zero(), B::Zero()};
                const V shift = patch.otherPoint - patch.point;
                const double area = patch.areaVector.dot(velocity);
                for (const auto &tri : patch.triangles)
                {
                    std::array<V, 3> other{tri[0] + shift, tri[1] + shift, tri[2] + shift};
                    integral[0] += 0.5 * area * TriangleMeanBasis(tri, patch.owner);
                    integral[1] += 0.5 * area * TriangleMeanBasis(other, patch.partner);
                }
                integral[0] -= area * Basis(patch.point, _x[patch.owner], _ref[patch.owner], _moments[patch.owner]);
                integral[1] -= area * Basis(patch.otherPoint, _x[patch.partner], _ref[patch.partner], _moments[patch.partner]);
                _patchMomentOperators.push_back(integral);
            }
        }

        template <class Matrix>
        void Sync(std::vector<Matrix> &values, int entries)
        {
            DNDS_check_throw_info(sizeof(Matrix) == entries * sizeof(double), "Unexpected Eigen storage padding");
            const int kind = entries == 5 ? 0 : 1;
            auto &send = _haloSendBuffer[kind];
            auto &receive = _haloRecvBuffer[kind];
            for (std::size_t j = 0; j < _haloSendNodes.size(); j++)
                std::copy_n(values[_haloSendNodes[j]].data(), entries, send.data() + entries * j);
            MPI_Alltoallv(send.data(), _haloSendCounts[kind].data(), _haloSendOffsets[kind].data(), MPI_DOUBLE,
                          receive.data(), _haloRecvCounts[kind].data(), _haloRecvOffsets[kind].data(), MPI_DOUBLE, MPI_COMM_WORLD);
            for (std::size_t j = 0; j < _haloRecvNodes.size(); j++)
                std::copy_n(receive.data() + entries * j, entries, values[_haloRecvNodes[j]].data());
        }

        void BuildHalo()
        {
            for (const auto &row : _operators)
                for (const auto &entry : row)
                    if (!Owned(entry.neighbor)) _haloRecvNodes.push_back(entry.neighbor);
            std::sort(_haloRecvNodes.begin(), _haloRecvNodes.end());
            _haloRecvNodes.erase(std::unique(_haloRecvNodes.begin(), _haloRecvNodes.end()), _haloRecvNodes.end());
            std::vector<int> recvCounts(_ranks, 0), sendCounts(_ranks, 0), recvOffsets(_ranks, 0), sendOffsets(_ranks, 0);
            for (int node : _haloRecvNodes)
            {
                const int owner = std::upper_bound(_offsets5.begin(), _offsets5.end(), 5 * node) - _offsets5.begin() - 1;
                ++recvCounts[owner];
            }
            MPI_Alltoall(recvCounts.data(), 1, MPI_INT, sendCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            for (int r = 1; r < _ranks; r++)
            {
                recvOffsets[r] = recvOffsets[r - 1] + recvCounts[r - 1];
                sendOffsets[r] = sendOffsets[r - 1] + sendCounts[r - 1];
            }
            _haloSendNodes.resize(sendOffsets.back() + sendCounts.back());
            MPI_Alltoallv(_haloRecvNodes.data(), recvCounts.data(), recvOffsets.data(), MPI_INT,
                          _haloSendNodes.data(), sendCounts.data(), sendOffsets.data(), MPI_INT, MPI_COMM_WORLD);
            for (int node : _haloSendNodes) DNDS_check_throw_info(Owned(node), "Invalid reference halo owner request");
            for (int kind = 0; kind < 2; kind++)
            {
                const int entries = kind == 0 ? 5 : 45;
                for (int r = 0; r < _ranks; r++)
                {
                    _haloSendCounts[kind].push_back(entries * sendCounts[r]);
                    _haloRecvCounts[kind].push_back(entries * recvCounts[r]);
                    _haloSendOffsets[kind].push_back(entries * sendOffsets[r]);
                    _haloRecvOffsets[kind].push_back(entries * recvOffsets[r]);
                }
                _haloSendBuffer[kind].resize(entries * _haloSendNodes.size());
                _haloRecvBuffer[kind].resize(entries * _haloRecvNodes.size());
            }
        }

        V FaceCenter(const std::vector<int> &nodes) const
        {
            V result = V::Zero();
            double sum = 0;
            for (int j = 0; j < static_cast<int>(nodes.size()); j++)
            {
                double weight = 1;
                if (nodes.size() == 4)
                    weight = (_x[nodes[(j + 1) % 4]] - _x[nodes[j]]).squaredNorm() +
                             (_x[nodes[(j + 3) % 4]] - _x[nodes[j]]).squaredNorm();
                result += weight * _x[nodes[j]];
                sum += weight;
            }
            return result / sum;
        }

        void SurfaceMoment(int node, const std::array<V, 3> &triangle, const V &s)
        {
            if (!Owned(node))
                return;
            const double a = 0.091576213509771, b = 0.8168475729804585;
            const double c = 0.445948490915965, d = 0.108103018168070;
            const std::array<std::pair<Eigen::Vector2d, double>, 6> rule{{
                {Eigen::Vector2d(a, a), 0.109951743655322}, {Eigen::Vector2d(b, a), 0.109951743655322},
                {Eigen::Vector2d(a, b), 0.109951743655322}, {Eigen::Vector2d(c, c), 0.2233815896780115},
                {Eigen::Vector2d(d, c), 0.2233815896780115}, {Eigen::Vector2d(c, d), 0.2233815896780115}}};
            for (const auto &[point, weight] : rule)
            {
                const V delta = triangle[0] + point(0) * (triangle[1] - triangle[0]) +
                                point(1) * (triangle[2] - triangle[0]) - _x[node];
                const V xi = delta.cwiseQuotient(_ref[node]);
                B contribution;
                contribution << delta(0) * xi(0) * s(0) / 2,
                    delta(1) * xi(1) * s(1) / 2, delta(2) * xi(2) * s(2) / 2,
                    delta(0) * xi(0) * xi(0) * s(0) / 3,
                    xi(0) * xi(1) * (delta(0) * s(0) + delta(1) * s(1)) / 4,
                    delta(1) * xi(1) * xi(1) * s(1) / 3,
                    xi(1) * xi(2) * (delta(1) * s(1) + delta(2) * s(2)) / 4,
                    delta(2) * xi(2) * xi(2) * s(2) / 3,
                    xi(0) * xi(2) * (delta(0) * s(0) + delta(2) * s(2)) / 4;
                _moments[node] += weight * contribution;
            }
        }

        void AddTet(int node, const V &midpoint, const V &face, const V &cell, const V &s)
        {
            const double measure = std::abs((midpoint - _x[node]).dot(s)) / 3;
            _tetVolume[node] += measure;
            if (Owned(node))
                _tets[node - _first].push_back({{midpoint, face, cell, _x[node]}, measure});
        }

        void Read(const std::string &path)
        {
            int file = 0;
            Cgns(cg_open(path.c_str(), CG_MODE_READ, &file));
            int zones = 0;
            Cgns(cg_nzones(file, 1, &zones));
            DNDS_check_throw_info(zones == 1, "Complete reference probe requires one unstructured zone");
            cgsize_t size[3]{};
            char name[33]{};
            Cgns(cg_zone_read(file, 1, 1, name, size));
            _x.resize(size[0]);
            cgsize_t lower = 1, upper = size[0];
            for (int dim = 0; dim < 3; dim++)
            {
                std::vector<double> coordinate(size[0]);
                const char *names[3]{"CoordinateX", "CoordinateY", "CoordinateZ"};
                Cgns(cg_coord_read(file, 1, 1, names[dim], RealDouble, &lower, &upper, coordinate.data()));
                for (int node = 0; node < static_cast<int>(_x.size()); node++)
                    _x[node](dim) = coordinate[node];
            }
            int sections = 0;
            Cgns(cg_nsections(file, 1, 1, &sections));
            for (int section = 1; section <= sections; section++)
            {
                ElementType_t type;
                cgsize_t start = 0, end = 0;
                int boundary = 0, parent = 0;
                Cgns(cg_section_read(file, 1, 1, section, name, &type, &start, &end, &boundary, &parent));
                if (type != PENTA_6)
                    continue;
                std::vector<cgsize_t> nodes(6 * (end - start + 1));
                Cgns(cg_elements_read(file, 1, 1, section, nodes.data(), nullptr));
                for (std::size_t offset = 0; offset < nodes.size(); offset += 6)
                {
                    std::array<int, 6> cell;
                    for (int j = 0; j < 6; j++)
                        cell[j] = static_cast<int>(nodes[offset + j] - 1);
                    _cells.push_back(cell);
                }
            }
            Cgns(cg_close(file));
            DNDS_check_throw_info(_cells.size() == static_cast<std::size_t>(size[1]), "Mesh is not entirely PRISM_6");
            _first = static_cast<int>(_x.size()) * _rank / _ranks;
            _last = static_cast<int>(_x.size()) * (_rank + 1) / _ranks;
            DNDS_check_throw_info(_last > _first, "Too many ranks for the reference mesh");
            for (int r = 0; r < _ranks; r++)
            {
                const int lo = static_cast<int>(_x.size()) * r / _ranks;
                const int hi = static_cast<int>(_x.size()) * (r + 1) / _ranks;
                _counts5.push_back(5 * (hi - lo)); _offsets5.push_back(5 * lo);
                _counts45.push_back(45 * (hi - lo)); _offsets45.push_back(45 * lo);
            }
        }

        void Geometry()
        {
            const std::array<std::vector<int>, 5> localFaces{{{0, 2, 1}, {3, 4, 5},
                {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}}};
            std::map<std::array<int, 4>, int> faceMap;
            std::unordered_map<unsigned long long, int> edgeMap;
            const auto edgeKey = [](int i, int j)
            {
                if (i > j) std::swap(i, j);
                return (static_cast<unsigned long long>(i) << 32) | static_cast<unsigned int>(j);
            };
            _cellFaces.resize(_cells.size());
            for (int cell = 0; cell < static_cast<int>(_cells.size()); cell++)
                for (int f = 0; f < 5; f++)
                {
                    std::vector<int> nodes;
                    for (int v : localFaces[f]) nodes.push_back(_cells[cell][v]);
                    std::array<int, 4> key{-1, -1, -1, -1};
                    std::copy(nodes.begin(), nodes.end(), key.begin());
                    std::sort(key.begin(), key.end());
                    auto [position, inserted] = faceMap.emplace(key, static_cast<int>(_faces.size()));
                    if (inserted) _faces.push_back({nodes, {}, FaceCenter(nodes)});
                    const int id = position->second;
                    _faces[id].cells.push_back(cell);
                    _cellFaces[cell][f] = id;
                }
            faceMap.clear();
            _incidence.resize(_x.size());
            for (const auto &face : _faces)
                for (int j = 0; j < static_cast<int>(face.nodes.size()); j++)
                {
                    const int left = std::min(face.nodes[j], face.nodes[(j + 1) % face.nodes.size()]);
                    const int right = std::max(face.nodes[j], face.nodes[(j + 1) % face.nodes.size()]);
                    auto [position, inserted] = edgeMap.emplace(edgeKey(left, right), static_cast<int>(_edges.size()));
                    if (inserted)
                    {
                        _edges.push_back({left, right, 0.5 * (_x[left] + _x[right]), V::Zero()});
                        _incidence[left].push_back({position->second, 1});
                        _incidence[right].push_back({position->second, -1});
                    }
                }
            _ref.resize(_x.size());
            _moments.assign(_x.size(), B::Zero());
            _volume.assign(_x.size(), 0); _tetVolume.assign(_x.size(), 0);
            _tets.resize(_last - _first); _samples.resize(_last - _first);
            for (int node = 0; node < static_cast<int>(_x.size()); node++)
            {
                V lower = _x[node], upper = _x[node];
                for (const auto &[edge, sign] : _incidence[node])
                {
                    lower = lower.cwiseMin(_edges[edge].midpoint);
                    upper = upper.cwiseMax(_edges[edge].midpoint);
                }
                _ref[node] = 0.5 * (upper - lower);
                DNDS_check_throw_info(_ref[node].minCoeff() > 0, "Invalid source midpoint reference lengths");
            }
            if (_keepSurfacePieces) _surfacePieces.resize(_edges.size());
            _cellCenters.resize(_cells.size());
            for (int cell = 0; cell < static_cast<int>(_cells.size()); cell++)
            {
                V center = V::Zero();
                double weightSum = 0;
                for (int id : _cellFaces[cell])
                {
                    const auto &face = _faces[id];
                    double area2 = 0;
                    for (int j = 0; j < static_cast<int>(face.nodes.size()); j++)
                        area2 += (_x[face.nodes[j]] - face.center).cross(
                            _x[face.nodes[(j + 1) % face.nodes.size()]] - face.center).norm();
                    center += area2 * area2 * face.center;
                    weightSum += area2 * area2;
                }
                center /= weightSum;
                _cellCenters[cell] = center;
                for (int id : _cellFaces[cell])
                {
                    const auto &face = _faces[id];
                    for (int j = 0; j < static_cast<int>(face.nodes.size()); j++)
                    {
                        const int edgeId = edgeMap.at(edgeKey(face.nodes[j], face.nodes[(j + 1) % face.nodes.size()]));
                        auto &edge = _edges[edgeId];
                        V s = 0.5 * (face.center - edge.midpoint).cross(center - edge.midpoint);
                        if (s.dot(_x[edge.right] - _x[edge.left]) < 0) s = -s;
                        edge.areaVector += s;
                        if (_keepSurfacePieces && (Owned(edge.left) || Owned(edge.right)))
                            _surfacePieces[edgeId].push_back({{edge.midpoint, face.center, center}, s});
                        AddTet(edge.left, edge.midpoint, face.center, center, s);
                        AddTet(edge.right, edge.midpoint, face.center, center, s);
                        SurfaceMoment(edge.left, {edge.midpoint, face.center, center}, s);
                        SurfaceMoment(edge.right, {edge.midpoint, face.center, center}, -s);
                    }
                }
            }
            edgeMap.clear();
            std::map<std::array<long long, 3>, int> coordinates;
            const auto coordKey = [](const V &value)
            {
                return std::array<long long, 3>{std::llround(value(0) * 1e8),
                    std::llround(value(1) * 1e8), std::llround(value(2) * 1e8)};
            };
            for (int node = 0; node < static_cast<int>(_x.size()); node++)
                DNDS_check_throw_info(coordinates.emplace(coordKey(_x[node]), node).second, "Duplicate raw node coordinates");
            std::vector<V> boundaryArea(_x.size(), V::Zero());
            for (const auto &face : _faces)
            {
                DNDS_check_throw_info(face.cells.size() <= 2, "Non-manifold primal face");
                if (face.cells.size() != 1) continue;
                V normal = (_x[face.nodes[1]] - _x[face.nodes[0]]).cross(_x[face.nodes[2]] - _x[face.nodes[0]]).normalized();
                if (normal.dot(face.center - _cellCenters[face.cells[0]]) < 0) normal = -normal;
                int axis = 0; normal.cwiseAbs().maxCoeff(&axis);
                DNDS_check_throw_info(std::abs(normal(axis)) > 1 - 1e-10, "Reference probe supports box periodic boundaries only");
                V shift = V::Zero(); shift(axis) = -std::copysign(domain(axis), normal(axis));
                for (int j = 0; j < static_cast<int>(face.nodes.size()); j++)
                {
                    const int node = face.nodes[j];
                    const V m1 = 0.5 * (_x[node] + _x[face.nodes[(j + 1) % face.nodes.size()]]);
                    const V m2 = 0.5 * (_x[node] + _x[face.nodes[(j + face.nodes.size() - 1) % face.nodes.size()]]);
                    const double area = 0.5 * (face.center - _x[node]).cross(m1 - _x[node]).norm() +
                                        0.5 * (face.center - _x[node]).cross(m2 - _x[node]).norm();
                    const V point = 0.25 * (_x[node] + face.center + m1 + m2);
                    const auto partner = coordinates.find(coordKey(_x[node] + shift));
                    DNDS_check_throw_info(partner != coordinates.end(), "Cannot pair a periodic boundary node");
                    const V s = area * normal;
                    boundaryArea[node] += s;
                    SurfaceMoment(node, {_x[node], face.center, m1}, 0.5 * s);
                    SurfaceMoment(node, {_x[node], face.center, m2}, 0.5 * s);
                    if (Owned(node))
                    {
                        Patch patch{node, partner->second, point, point + shift, s};
                        patch.triangles[0] = {_x[node], face.center, m1};
                        patch.triangles[1] = {_x[node], face.center, m2};
                        _patches.push_back(patch);
                        _samples[node - _first].push_back({partner->second, point, point + shift, 2 * std::sqrt(area)});
                    }
                }
            }
            for (const auto &edge : _edges)
            {
                const double v = edge.areaVector.dot(_x[edge.right] - _x[edge.left]) / 6;
                DNDS_check_throw_info(v > 0, "Nonpositive directed edge-volume contribution");
                _volume[edge.left] += v; _volume[edge.right] += v;
            }
            const auto rule = TetRule();
            for (int node = _first; node < _last; node++)
            {
                V closure = boundaryArea[node];
                double areaSum = boundaryArea[node].norm();
                std::vector<Sample> ordered;
                for (const auto &[edgeId, sign] : _incidence[node])
                {
                    const auto &edge = _edges[edgeId];
                    closure += sign * edge.areaVector;
                    areaSum += edge.areaVector.norm();
                    ordered.push_back({sign > 0 ? edge.right : edge.left,
                        edge.midpoint, edge.midpoint, (_x[edge.right] - _x[edge.left]).norm()});
                }
                ordered.insert(ordered.end(), _samples[node - _first].begin(), _samples[node - _first].end());
                _samples[node - _first] = std::move(ordered);
                _moments[node] /= _volume[node];
                B direct = B::Zero();
                for (const auto &tet : _tets[node - _first])
                    for (const auto &[p, w] : rule)
                    {
                        const V point = tet.points[0] + p(0) * (tet.points[1] - tet.points[0]) +
                            p(1) * (tet.points[2] - tet.points[0]) + p(2) * (tet.points[3] - tet.points[0]);
                        direct += tet.volume * w / _volume[node] * Basis(point, _x[node], _ref[node], B::Zero());
                    }
                _maximumMomentMismatch = std::max(_maximumMomentMismatch, (direct - _moments[node]).cwiseAbs().maxCoeff());
                _maximumVolumeMismatch = std::max(_maximumVolumeMismatch, std::abs(_tetVolume[node] / _volume[node] - 1));
                _maximumClosure = std::max(_maximumClosure, closure.norm() / areaSum);
                _totalVolume += _volume[node];
            }
            std::vector<int> counts9, offsets9;
            for (int r = 0; r < _ranks; r++)
            {
                counts9.push_back(_counts5[r] / 5 * 9); offsets9.push_back(_offsets5[r] / 5 * 9);
            }
            DNDS_check_throw_info(sizeof(B) == 9 * sizeof(double), "Unexpected basis storage padding");
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, _moments[0].data(), counts9.data(), offsets9.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            DNDS_check_throw_info(_maximumVolumeMismatch < 1e-10 && _maximumClosure < 1e-10 &&
                _maximumMomentMismatch < 1e-9, "Reference geometry validation failed");
            _faces.clear(); _faces.shrink_to_fit(); _cells.clear(); _cells.shrink_to_fit();
            _cellFaces.clear(); _cellFaces.shrink_to_fit(); _cellCenters.clear(); _cellCenters.shrink_to_fit();
        }

        void BuildOperators()
        {
            _operators.resize(_last - _first); _matrices.resize(_last - _first);
            _coefficients.assign(_x.size(), C::Zero()); _state.assign(_x.size(), U::Zero());
            for (int node = _first; node < _last; node++)
            {
                A matrix = A::Zero();
                for (const auto &sample : _samples[node - _first])
                {
                    const Rows left = DifferentialRows(sample.point, _x[node], _ref[node], _moments[node], sample.length);
                    matrix.noalias() += left.transpose() * left;
                }
                Eigen::SelfAdjointEigenSolver<A> spectrum(matrix);
                DNDS_check_throw_info(spectrum.info() == Eigen::Success && spectrum.eigenvalues()(0) > 0,
                    "Singular reference local reconstruction matrix");
                _maximumCondition = std::max(_maximumCondition, spectrum.eigenvalues()(8) / spectrum.eigenvalues()(0));
                Eigen::PartialPivLU<A> lu(matrix);
                _matrices[node - _first] = matrix;
                for (const auto &sample : _samples[node - _first])
                {
                    const Rows left = DifferentialRows(sample.point, _x[node], _ref[node], _moments[node], sample.length);
                    const Rows right = DifferentialRows(sample.otherPoint, _x[sample.neighbor], _ref[sample.neighbor], _moments[sample.neighbor], sample.length);
                    const A cross = left.transpose() * right;
                    const B jump = left.row(0).transpose();
                    Coupling entry{sample.neighbor, lu.solve(cross), lu.solve(jump)};
                    _maximumMatrixSolveError = std::max(_maximumMatrixSolveError,
                        (matrix * entry.matrix - cross).norm() / std::max(1.0, cross.norm()));
                    _operators[node - _first].push_back(std::move(entry));
                }
            }
            if (_rank == 0)
            {
                _audit["first_node"] = {{"id", _first}, {"center", Array(_x[_first])},
                    {"ref", Array(_ref[_first])}, {"moment", Array(_moments[_first])}, {"A", Array(_matrices[0])}};
                _audit["first_node"]["samples"] = nlohmann::json::array();
                for (const auto &sample : _samples[0])
                    _audit["first_node"]["samples"].push_back({{"neighbor", sample.neighbor},
                        {"point", Array(sample.point)}, {"other_point", Array(sample.otherPoint)},
                        {"length", sample.length}, {"other_center", Array(_x[sample.neighbor])},
                        {"other_ref", Array(_ref[sample.neighbor])}, {"other_moment", Array(_moments[sample.neighbor])}});
            }
        }

        double CoupledResidual() const
        {
            double sums[2]{0, 0}, global[2]{};
            for (int node = _first; node < _last; node++)
            {
                C right = C::Zero();
                for (const auto &entry : _operators[node - _first])
                    right.noalias() += entry.matrix * _coefficients[entry.neighbor] +
                        entry.jump * (_state[entry.neighbor] - _state[node]).transpose();
                const A &matrix = _matrices[node - _first];
                sums[0] += (matrix * (_coefficients[node] - right)).squaredNorm();
                sums[1] += (matrix * right).squaredNorm();
            }
            MPI_Allreduce(sums, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return std::sqrt(global[0] / std::max(global[1], 1e-300));
        }

        void Reconstruct(int sweeps)
        {
            for (int sweep = 0; sweep < sweeps; sweep++)
            {
                for (int node = _first; node < _last; node++)
                {
                    C next = C::Zero();
                    for (const auto &entry : _operators[node - _first])
                    {
                        next.noalias() += entry.matrix * _coefficients[entry.neighbor];
                        next.noalias() += entry.jump * (_state[entry.neighbor] - _state[node]).transpose();
                    }
                    DNDS_check_throw_info(next.allFinite(), "Nonfinite reference reconstruction");
                    _coefficients[node] = next;
                }
                Sync(_coefficients, 45);
                if (_rhsCalls == 0 && sweep == 0 && _rank == 0)
                    _audit["first_node"]["coefficients_after_first_sweep"] = Array(_coefficients[_first]);
            }
            _lastReconstructionResidual = CoupledResidual();
            if (_rhsCalls == 0)
            {
                _firstReconstructionResidual = _lastReconstructionResidual;
                if (_rank == 0)
                {
                    _audit["first_node"]["initial_mean"] = Array(_state[_first]);
                    _audit["first_node"]["coefficients_after_initial_reconstruction"] = Array(_coefficients[_first]);
                    for (auto &sample : _audit["first_node"]["samples"])
                        sample["neighbor_mean"] = Array(_state[sample["neighbor"].get<int>()]);
                }
            }
        }

        void BuildQuadratureFaces()
        {
            std::size_t count = 12 * _patches.size();
            for (const auto &pieces : _surfacePieces) count += 6 * pieces.size();
            _quadratureFaces.reserve(count);
            const auto add = [&](int left, int right, bool internal, const V &point, const V &other,
                                 const V &normal, double weight)
            {
                if (_quadratureFaces.empty()) _firstQuadraturePoint = point;
                QuadratureFace face;
                face.left = left; face.right = right; face.internal = internal;
                face.normal = normal; face.weight = weight;
                face.leftBasis = Basis(point, _x[left], _ref[left], _moments[left]);
                face.rightBasis = Basis(other, _x[right], _ref[right], _moments[right]);
                _quadratureFaces.push_back(face);
            };
            for (std::size_t id = 0; id < _edges.size(); id++)
            {
                const auto &edge = _edges[id];
                for (const auto &piece : _surfacePieces[id])
                    for (const auto &[p, w] : TriangleRule())
                    {
                        const V point = piece.points[0] + p(0) * (piece.points[1] - piece.points[0]) +
                            p(1) * (piece.points[2] - piece.points[0]);
                        add(edge.left, edge.right, true, point, point, piece.areaVector.normalized(), piece.areaVector.norm() * w);
                    }
            }
            for (const auto &patch : _patches)
                for (const auto &triangle : patch.triangles)
                    for (const auto &[p, w] : TriangleRule())
                    {
                        const V point = triangle[0] + p(0) * (triangle[1] - triangle[0]) + p(1) * (triangle[2] - triangle[0]);
                        add(patch.owner, patch.partner, false, point, point + patch.otherPoint - patch.point,
                            patch.areaVector.normalized(), 0.5 * patch.areaVector.norm() * w);
                    }
        }

        U FaceState(int node, const B &basis)
        {
            const U reconstructed = _state[node] + _coefficients[node].transpose() * basis;
            if (Physical(reconstructed)) return reconstructed;
            ++_fallbackCount;
            return _state[node];
        }

        U FaceState(int node, const V &point)
        {
            if (_analyticFaces) return Exact(point, 0, _problem);
            U reconstructed = _state[node] + _coefficients[node].transpose() * Basis(point, _x[node], _ref[node], _moments[node]);
            if (Physical(reconstructed)) return reconstructed;
            ++_fallbackCount;
            return _state[node];
        }

        std::vector<U> RHS(int sweeps)
        {
            if (!_skipReconstruction) Reconstruct(sweeps);
            std::vector<U> rhs(_last - _first, U::Zero());
            long long calls = 0;
            if (_quadratureFlux && !_quadratureFaces.empty())
            {
                for (const auto &face : _quadratureFaces)
                {
                    const U left = FaceState(face.left, face.leftBasis), right = FaceState(face.right, face.rightBasis);
                    const U flux = Roe(left, right, face.normal);
                    if (Owned(face.left)) rhs[face.left - _first] -= face.weight * (flux - PhysicalFlux(_state[face.left], face.normal));
                    if (face.internal && Owned(face.right)) rhs[face.right - _first] += face.weight * (flux - PhysicalFlux(_state[face.right], face.normal));
                    if (_rhsCalls == 0 && calls == 0 && _rank == 0)
                        _audit["first_edge"] = {{"left", face.left}, {"right", face.right},
                            {"point", Array(_firstQuadraturePoint)}, {"area_vector", Array(face.weight * face.normal)},
                            {"left_center", Array(_x[face.left])}, {"right_center", Array(_x[face.right])},
                            {"left_ref", Array(_ref[face.left])}, {"right_ref", Array(_ref[face.right])},
                            {"left_moment", Array(_moments[face.left])}, {"right_moment", Array(_moments[face.right])},
                            {"left_coefficients", Array(_coefficients[face.left])}, {"right_coefficients", Array(_coefficients[face.right])},
                            {"left_mean", Array(_state[face.left])}, {"right_mean", Array(_state[face.right])},
                            {"left_state", Array(left)}, {"right_state", Array(right)}, {"flux", Array(flux)},
                            {"surface_quadrature_sample", true}};
                    ++calls;
                }
                _faceCalls = calls;
                for (int node = _first; node < _last; node++) rhs[node - _first] /= _volume[node];
                ++_rhsCalls;
                return rhs;
            }
            for (const auto &edge : _edges)
            {
                if (!Owned(edge.left) && !Owned(edge.right)) continue;
                const double area = edge.areaVector.norm();
                const V n = edge.areaVector / area;
                U left = U::Zero(), right = U::Zero(), flux = U::Zero();
                if (!_quadratureFlux)
                {
                    left = FaceState(edge.left, edge.midpoint); right = FaceState(edge.right, edge.midpoint);
                    flux = Roe(left, right, n);
                    if (Owned(edge.left)) rhs[edge.left - _first] -= area * (flux - PhysicalFlux(_state[edge.left], n));
                    if (Owned(edge.right)) rhs[edge.right - _first] += area * (flux - PhysicalFlux(_state[edge.right], n));
                    if (_entropyMomentCorrection && !_analyticFaces)
                    {
                        const std::size_t id = &edge - _edges.data();
                        const auto &moment = _edgeMomentOperators[_edgeMomentIndex[id]];
                        const double correction = 0.5 * (_coefficients[edge.left].col(0).dot(moment[0]) +
                                                        _coefficients[edge.right].col(0).dot(moment[1]));
                        U scale; scale << 1, 1, 1, 0, 1;
                        if (Owned(edge.left)) rhs[edge.left - _first] -= correction * scale;
                        if (Owned(edge.right)) rhs[edge.right - _first] += correction * scale;
                    }
                    ++calls;
                }
                else
                {
                    const std::size_t id = &edge - _edges.data();
                    for (const auto &piece : _surfacePieces[id])
                        for (const auto &[p, w] : TriangleRule())
                        {
                            const V point = piece.points[0] + p(0) * (piece.points[1] - piece.points[0]) +
                                            p(1) * (piece.points[2] - piece.points[0]);
                            const double weight = piece.areaVector.norm() * w;
                            const V normal = piece.areaVector.normalized();
                            const U value = Roe(FaceState(edge.left, point), FaceState(edge.right, point), normal);
                            if (Owned(edge.left)) rhs[edge.left - _first] -= weight * (value - PhysicalFlux(_state[edge.left], normal));
                            if (Owned(edge.right)) rhs[edge.right - _first] += weight * (value - PhysicalFlux(_state[edge.right], normal));
                            ++calls;
                        }
                }
                if (_rhsCalls == 0 && _rank == 0 && calls == 1 && !_quadratureFlux)
                    _audit["first_edge"] = {{"left", edge.left}, {"right", edge.right},
                        {"point", Array(edge.midpoint)}, {"area_vector", Array(edge.areaVector)},
                        {"left_center", Array(_x[edge.left])}, {"right_center", Array(_x[edge.right])},
                        {"left_ref", Array(_ref[edge.left])}, {"right_ref", Array(_ref[edge.right])},
                        {"left_moment", Array(_moments[edge.left])}, {"right_moment", Array(_moments[edge.right])},
                        {"left_coefficients", Array(_coefficients[edge.left])}, {"right_coefficients", Array(_coefficients[edge.right])},
                        {"left_mean", Array(_state[edge.left])}, {"right_mean", Array(_state[edge.right])},
                        {"left_state", Array(left)}, {"right_state", Array(right)}, {"flux", Array(flux)}};
            }
            for (const auto &patch : _patches)
            {
                const double area = patch.areaVector.norm();
                const V n = patch.areaVector / area;
                if (!_quadratureFlux)
                {
                    const U left = FaceState(patch.owner, patch.point), right = FaceState(patch.partner, patch.otherPoint);
                    rhs[patch.owner - _first] -= area * (Roe(left, right, n) - PhysicalFlux(_state[patch.owner], n));
                    if (_entropyMomentCorrection && !_analyticFaces)
                    {
                        const std::size_t id = &patch - _patches.data();
                        const auto &moment = _patchMomentOperators[id];
                        const double correction = 0.5 * (_coefficients[patch.owner].col(0).dot(moment[0]) +
                                                        _coefficients[patch.partner].col(0).dot(moment[1]));
                        U scale; scale << 1, 1, 1, 0, 1;
                        rhs[patch.owner - _first] -= correction * scale;
                    }
                    ++calls;
                }
                else
                    for (const auto &triangle : patch.triangles)
                        for (const auto &[p, w] : TriangleRule())
                        {
                            const V point = triangle[0] + p(0) * (triangle[1] - triangle[0]) + p(1) * (triangle[2] - triangle[0]);
                            const V other = point + patch.otherPoint - patch.point;
                            rhs[patch.owner - _first] -= 0.5 * area * w *
                                (Roe(FaceState(patch.owner, point), FaceState(patch.partner, other), n) - PhysicalFlux(_state[patch.owner], n));
                            ++calls;
                        }
            }
            _faceCalls = calls;
            for (int node = _first; node < _last; node++) rhs[node - _first] /= _volume[node];
            ++_rhsCalls;
            return rhs;
        }

        U Exact(const V &point, double time, const std::string &problem) const
        {
            if (problem == "entropy2d")
                return Pack(1 + 0.2 * std::sin(point(0) + point(1) - 1.5 * time), V(1, 0.5, 0), 1);
            if (problem == "tgv2d")
            {
                const V velocity(std::sin(point(0)) * std::cos(point(1)),
                                 -std::cos(point(0)) * std::sin(point(1)), 0);
                const double pressure = 1 / (gammaGas * 0.09) +
                    0.25 * (std::cos(2 * point(0)) + std::cos(2 * point(1)));
                return Pack(1, velocity, pressure);
            }
            if (problem == "tgv")
            {
                const V phase = 2 * pi * point.cwiseQuotient(domain);
                const V velocity(std::sin(phase(0)) * std::cos(phase(1)) * std::cos(phase(2)),
                                 -std::cos(phase(0)) * std::sin(phase(1)) * std::cos(phase(2)), 0);
                const double pressure = 1 / (gammaGas * 0.01) +
                    (std::cos(2 * phase(0)) + std::cos(2 * phase(1))) * (2 + std::cos(2 * phase(2))) / 16;
                return Pack(gammaGas * 0.01 * pressure, velocity, pressure);
            }
            if (problem == "entropy" || problem == "entropy_xy")
            {
                const double phase = 2 * pi * ((point(0) - time) / 10 + (point(1) - time) / 10 +
                    (problem == "entropy" ? point(2) / 4 : 0));
                return Pack(1 + 0.2 * std::sin(phase), V(1, 1, 0), 1);
            }
            V advected = point;
            for (int d = 0; d < 2; d++)
            {
                advected(d) = std::fmod(point(d) - time, 10);
                if (advected(d) < 0) advected(d) += 10;
            }
            const double dx = advected(0) - 5, dy = advected(1) - 5;
            const double r2 = dx * dx + dy * dy, beta = 5 / (2 * pi) * std::exp((1 - r2) / 2);
            const double theta = 1 - (gammaGas - 1) * 25 / (8 * gammaGas * pi * pi) * std::exp(1 - r2);
            const double rho = std::pow(theta, 1 / (gammaGas - 1));
            return Pack(rho, V(1 - beta * dy, 1 + beta * dx, 0), std::pow(rho, gammaGas));
        }

        U Mean(int node, double time, const std::string &problem, bool higher = false) const
        {
            U result = U::Zero();
            const auto rule = TetRule();
            DNDS::Geom::Elem::Quadrature reference({DNDS::Geom::Elem::Tet4}, 6);
            for (const auto &tet : _tets[node - _first])
                for (int q = 0; q < (higher ? reference.GetNumPoints() : 8); q++)
                {
                    V p;
                    double w;
                    if (higher)
                    {
                        const auto info = reference.GetQuadraturePointInfo(q);
                        p = std::get<0>(info); w = 6 * std::get<1>(info);
                    }
                    else { p = rule[q].first; w = rule[q].second; }
                    const V point = tet.points[0] + p(0) * (tet.points[1] - tet.points[0]) +
                        p(1) * (tet.points[2] - tet.points[0]) + p(2) * (tet.points[3] - tet.points[0]);
                    result += tet.volume * w / _volume[node] * Exact(point, time, problem);
                }
            return result;
        }

        int Region(int node) const
        {
            if (std::abs(_x[node](2)) < 1e-8 || std::abs(_x[node](2) - 4) < 1e-8) return 1;
            if (std::abs(_x[node](0)) < 1e-8 || std::abs(_x[node](0) - 10) < 1e-8 ||
                std::abs(_x[node](1)) < 1e-8 || std::abs(_x[node](1) - 10) < 1e-8) return 2;
            return 3;
        }

        nlohmann::json ErrorNorms(const std::vector<double> &error) const
        {
            double local[4][4]{}, global[4][4]{}, maxLocal[4]{}, maxGlobal[4]{};
            for (int node = _first; node < _last; node++)
                for (int region : {0, Region(node)})
                {
                    const double value = error[node - _first], v = _volume[node];
                    local[region][0] += v; local[region][1] += v * std::abs(value);
                    local[region][2] += v * value * value; local[region][3]++;
                    maxLocal[region] = std::max(maxLocal[region], std::abs(value));
                }
            MPI_Allreduce(local, global, 16, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(maxLocal, maxGlobal, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            const char *names[4]{"all", "z_boundary", "other_boundary", "interior"};
            nlohmann::json result;
            for (int r = 0; r < 4; r++)
                result[names[r]] = {{"volume", global[r][0]}, {"nodes", global[r][3]},
                    {"L1", global[r][1] / global[r][0]}, {"L2", std::sqrt(global[r][2] / global[r][0])},
                    {"Linf", maxGlobal[r]}, {"squared_error_integral", global[r][2]},
                    {"squared_error_share", global[r][2] / std::max(global[0][2], 1e-300)}};
            return result;
        }

        double ExactEntropyTimeDerivative(int node, const V &k) const
        {
            double result = 0;
            DNDS::Geom::Elem::Quadrature rule({DNDS::Geom::Elem::Tet4}, 6);
            for (const auto &tet : _tets[node - _first])
                for (int q = 0; q < rule.GetNumPoints(); q++)
                {
                    const auto [p, w] = rule.GetQuadraturePointInfo(q);
                    const V point = tet.points[0] + p(0) * (tet.points[1] - tet.points[0]) +
                        p(1) * (tet.points[2] - tet.points[0]) + p(2) * (tet.points[3] - tet.points[0]);
                    result += 6 * tet.volume * w / _volume[node] *
                        (-0.2 * (k(0) + k(1)) * std::cos(k.dot(point)));
                }
            return result;
        }

        std::vector<double> PolynomialFluxError(int polynomial, bool quadrature) const
        {
            const V velocity(1, 1, 0);
            std::vector<double> value(_last - _first, 0);
            const auto rho = [&](const V &point)
            {
                return polynomial < 3 ? point(polynomial) : point(0) * point(2);
            };
            for (std::size_t id = 0; id < _edges.size(); id++)
            {
                const auto &edge = _edges[id];
                if (!Owned(edge.left) && !Owned(edge.right)) continue;
                double flux = edge.areaVector.dot(velocity) * rho(edge.midpoint);
                if (quadrature)
                {
                    flux = 0;
                    for (const auto &piece : _surfacePieces[id])
                        for (const auto &[p, w] : TriangleRule())
                            flux += w * piece.areaVector.dot(velocity) * rho(piece.points[0] +
                                p(0) * (piece.points[1] - piece.points[0]) + p(1) * (piece.points[2] - piece.points[0]));
                }
                if (Owned(edge.left)) value[edge.left - _first] -= flux;
                if (Owned(edge.right)) value[edge.right - _first] += flux;
            }
            for (const auto &patch : _patches)
            {
                double flux = patch.areaVector.dot(velocity) * rho(patch.point);
                if (quadrature)
                {
                    flux = 0;
                    for (const auto &tri : patch.triangles)
                        for (const auto &[p, w] : TriangleRule())
                            flux += 0.5 * w * patch.areaVector.dot(velocity) * rho(tri[0] +
                                p(0) * (tri[1] - tri[0]) + p(1) * (tri[2] - tri[0]));
                }
                value[patch.owner - _first] -= flux;
            }
            for (int node = _first; node < _last; node++)
            {
                const double exact = polynomial < 3 ? -velocity(polynomial) :
                    -(_x[node](2) + _ref[node](2) * _moments[node](2));
                value[node - _first] = value[node - _first] / _volume[node] - exact;
            }
            return value;
        }

    public:
        Reference(const std::string &mesh)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &_rank); MPI_Comm_size(MPI_COMM_WORLD, &_ranks);
            _entropyMomentCorrection = std::getenv("HOCFV_ENTROPY_MOMENT_CORRECTION") != nullptr;
            _quadratureFlux = std::getenv("HOCFV_SURFACE_QUADRATURE") != nullptr;
            DNDS_check_throw_info(!_quadratureFlux || !_entropyMomentCorrection,
                                  "Surface quadrature cannot be combined with the entropy moment diagnostic");
            _keepSurfacePieces = std::getenv("HOCFV_AUDIT_RHS") != nullptr || _entropyMomentCorrection || _quadratureFlux;
            Read(mesh); Geometry(); BuildOperators();
            BuildHalo();
            if (_entropyMomentCorrection || std::getenv("HOCFV_AUDIT_RHS")) BuildEntropyMomentOperators();
            if (_quadratureFlux && !std::getenv("HOCFV_AUDIT_RHS") && !std::getenv("HOCFV_UNCACHED_QUADRATURE"))
                BuildQuadratureFaces();
            if (!std::getenv("HOCFV_AUDIT_RHS") && (!_quadratureFlux || !_quadratureFaces.empty()))
            {
                _surfacePieces.clear(); _surfacePieces.shrink_to_fit();
            }
        }

        void AuditEntropyRHS(const std::string &problem, int sweeps, const std::string &output)
        {
            DNDS_check_throw_info(_keepSurfacePieces && (problem == "entropy" || problem == "entropy_xy"), "Invalid RHS audit setup");
            DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite RHS audit");
            _problem = problem;
            const V k(2 * pi / 10, 2 * pi / 10, problem == "entropy" ? 2 * pi / 4 : 0);
            std::vector<double> exact(_last - _first);
            for (int node = _first; node < _last; node++)
            {
                _state[node] = Mean(node, 0, problem);
                exact[node - _first] = ExactEntropyTimeDerivative(node, k);
            }
            Sync(_state, 5);
            Reconstruct(sweeps);
            const auto vr = _coefficients;
            const double vrResidual = _lastReconstructionResidual;
            for (int node = _first; node < _last; node++)
            {
                B density;
                const double phase = k.dot(_x[node]);
                for (int j = 0; j < 9; j++)
                {
                    if (j < 3) density(j) = 0.2 * std::cos(phase) * k(j) * _ref[node](j);
                    else
                    {
                        double value = -0.2 * std::sin(phase);
                        bool pure = false;
                        for (int d = 0; d < 3; d++)
                        {
                            if (powers[j][d] == 2) pure = true;
                            for (int order = 0; order < powers[j][d]; order++) value *= k(d) * _ref[node](d);
                        }
                        density(j) = pure ? 0.5 * value : value;
                    }
                }
                U scale; scale << 1, 1, 1, 0, 1;
                _coefficients[node] = density * scale.transpose();
            }
            Sync(_coefficients, 45);
            const auto taylor = _coefficients;
            _skipReconstruction = true;
            nlohmann::json summary = {{"problem", problem}, {"nodes", _x.size()}, {"ranks", _ranks},
                {"sweeps", sweeps}, {"time", 0}, {"vr_coupled_residual", vrResidual}};
            const char *names[7]{"vr_midpoint", "taylor_midpoint", "exact_midpoint", "vr_surface6", "taylor_surface6", "exact_surface6", "vr_moment_corrected"};
            for (int mode = 0; mode < 7; mode++)
            {
                _quadratureFlux = mode >= 3 && mode < 6;
                _entropyMomentCorrection = mode == 6;
                _analyticFaces = mode % 3 == 2;
                _coefficients = mode % 3 == 0 ? vr : taylor;
                const auto rhs = RHS(sweeps);
                std::vector<double> error(_last - _first);
                for (int node = _first; node < _last; node++) error[node - _first] = rhs[node - _first](0) - exact[node - _first];
                summary["schemes"][names[mode]] = ErrorNorms(error);
            }
            for (int p = 0; p < 4; p++)
                for (int quadrature = 0; quadrature < 2; quadrature++)
                    summary["polynomial_consistency"][p][quadrature ? "surface6" : "midpoint"] =
                        ErrorNorms(PolynomialFluxError(p, quadrature));
            if (_rank == 0)
            {
                std::ofstream file(output); file << std::setw(2) << summary << '\n';
                DNDS_check_throw_info(file.good(), "RHS audit write failed");
                std::cout << summary.dump(2) << std::endl;
            }
        }

        void DumpConservativeMeans(const std::string &output, const std::string &label, double time) const
        {
            std::ostringstream suffix;
            suffix << ".rank" << std::setw(4) << std::setfill('0') << _rank;
            std::ofstream field(output + "." + label + suffix.str() + ".csv");
            field << std::setprecision(17) << "id,x,y,z,volume,time,rho,rhou,rhov,rhow,rhoE\n";
            for (int node = _first; node < _last; node++)
            {
                field << node << ',' << _x[node](0) << ',' << _x[node](1) << ',' << _x[node](2)
                      << ',' << _volume[node] << ',' << time;
                for (int variable = 0; variable < 5; variable++) field << ',' << _state[node](variable);
                field << '\n';
            }
            DNDS_check_throw_info(field.good(), "Conservative mean field write failed");
            if (label != "initial") return;
            std::ofstream quadrature(output + ".quadrature" + suffix.str() + ".bin", std::ios::binary);
            quadrature.write("HOQv0001", 8);
            for (int node = _first; node < _last; node++)
                for (const auto &tet : _tets[node - _first])
                    for (const auto &[p, weight] : TetRule())
                    {
                        const V point = tet.points[0] + p(0) * (tet.points[1] - tet.points[0]) +
                            p(1) * (tet.points[2] - tet.points[0]) + p(2) * (tet.points[3] - tet.points[0]);
                        const double record[5]{static_cast<double>(node), point(0), point(1), point(2), tet.volume * weight};
                        quadrature.write(reinterpret_cast<const char *>(record), sizeof(record));
                    }
            DNDS_check_throw_info(quadrature.good(), "Conservative mean quadrature write failed");
        }

        void Run(const std::string &problem, double dt, double end, int sweeps, const std::string &output)
        {
            _problem = problem;
            DNDS_check_throw_info(!_entropyMomentCorrection || problem == "entropy" || problem == "entropy_xy",
                                  "Entropy moment diagnostic requires constant-velocity entropy flow");
            DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite reference result");
            const int steps = std::llround(end / dt);
            DNDS_check_throw_info(dt > 0 && steps > 0 && std::abs(steps * dt - end) < 1e-12, "Invalid fixed time input");
            U initial = U::Zero();
            for (int node = _first; node < _last; node++)
            {
                _state[node] = Mean(node, 0, problem);
                initial += _volume[node] * _state[node];
            }
            Sync(_state, 5);
            if (problem == "tgv2d" || problem == "entropy2d") DumpConservativeMeans(output, "initial", 0);
            const double start = MPI_Wtime();
            for (int step = 0; step < steps; step++)
            {
                const std::vector<U> base(_state.begin() + _first, _state.begin() + _last);
                for (int stage = 0; stage < 3; stage++)
                {
                    const auto rhs = RHS(sweeps);
                    const double c[3]{1, 0.25, 2.0 / 3.0};
                    if (step == 0 && _rank == 0)
                        _audit["rk_stage"][stage] = {{"base", Array(base[0])},
                            {"before", Array(_state[_first])}, {"rhs_divided_by_volume", Array(rhs[0])}, {"volume", _volume[_first]}};
                    for (int node = _first; node < _last; node++)
                    {
                        _state[node] = (1 - c[stage]) * base[node - _first] + c[stage] * (_state[node] + dt * rhs[node - _first]);
                        DNDS_check_throw_info(Physical(_state[node]), "Nonphysical RK mean in complete HOCFV reference");
                    }
                    Sync(_state, 5);
                    if (step == 0 && _rank == 0) _audit["rk_stage"][stage]["after"] = Array(_state[_first]);
                }
                if (_rank == 0 && (step + 1 == steps || (step + 1) % 8 == 0))
                    std::cout << "Step " << step + 1 << '/' << steps << ", coupled residual " << _lastReconstructionResidual << std::endl;
            }
            const double march = MPI_Wtime() - start;
            if (problem == "tgv2d" || problem == "entropy2d") DumpConservativeMeans(output, "final", end);
            const double lastStageResidual = _lastReconstructionResidual;
            double finalPointResidual = 0;
            if (problem == "tgv")
            {
                Reconstruct(sweeps);
                finalPointResidual = _lastReconstructionResidual;
                std::ostringstream filename;
                filename << output << ".rank" << std::setw(4) << std::setfill('0') << _rank << ".csv";
                std::ofstream field(filename.str());
                field << std::setprecision(17) << "x,y,z,rho,u,v,w,p,dual_volume\n";
                for (int node = _first; node < _last; node++)
                {
                    const U point = _state[node] + _coefficients[node].transpose() * Basis(_x[node], _x[node], _ref[node], _moments[node]);
                    DNDS_check_throw_info(Physical(point), "Nonphysical recovered Taylor-Green nodal state");
                    const V velocity = point.segment<3>(1) / point(0);
                    field << _x[node](0) << ',' << _x[node](1) << ',' << _x[node](2) << ',' << point(0)
                          << ',' << velocity(0) << ',' << velocity(1) << ',' << velocity(2)
                          << ',' << Pressure(point) << ',' << _volume[node] << '\n';
                }
                DNDS_check_throw_info(field.good(), "Taylor-Green field write failed");
            }
            std::ofstream meanField;
            if (std::getenv("HOCFV_WRITE_FIELDS") && problem != "tgv")
            {
                std::ostringstream filename;
                filename << output << ".mean.rank" << std::setw(4) << std::setfill('0') << _rank << ".csv";
                meanField.open(filename.str());
                meanField << std::setprecision(17) << "x,y,z,rho_numerical_mean,rho_exact_mean,dual_volume,distance_to_boundary\n";
            }
            double sums[8]{}, global[8]{};
            U final = U::Zero();
            double linf = 0, minimumDensity = 1e300, minimumPressure = 1e300;
            double maxReferenceDifference = 0;
            for (int node = _first; node < _last; node++)
            {
                const double exact = (problem == "tgv" || problem == "tgv2d") ? _state[node](0) : Mean(node, end, problem)(0);
                const double error = _state[node](0) - exact;
                const double higher = (problem == "tgv" || problem == "tgv2d") ? exact : Mean(node, end, problem, true)(0);
                const double highError = _state[node](0) - higher;
                sums[0] += _volume[node] * std::abs(error); sums[1] += _volume[node] * error * error;
                sums[2] += _volume[node] * highError * highError;
                sums[3] += _volume[node] * std::pow(exact - higher, 2);
                final += _volume[node] * _state[node];
                linf = std::max(linf, std::abs(error));
                minimumDensity = std::min(minimumDensity, _state[node](0));
                minimumPressure = std::min(minimumPressure, Pressure(_state[node]));
                maxReferenceDifference = std::max(maxReferenceDifference, std::abs(exact - higher));
                if (meanField.is_open())
                    meanField << _x[node](0) << ',' << _x[node](1) << ',' << _x[node](2) << ',' << _state[node](0)
                              << ',' << exact << ',' << _volume[node] << ','
                              << std::min(_x[node].minCoeff(), (domain - _x[node]).minCoeff()) << '\n';
            }
            MPI_Allreduce(sums, global, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double checks[7]{_maximumVolumeMismatch, _maximumClosure, _maximumMomentMismatch,
                _maximumCondition, _maximumMatrixSolveError, linf, maxReferenceDifference}, checksGlobal[7]{};
            MPI_Allreduce(checks, checksGlobal, 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            double vGlobal = 0, marchGlobal = 0, minimum[2]{minimumDensity, minimumPressure}, minimumGlobal[2]{};
            MPI_Allreduce(&_totalVolume, &vGlobal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&march, &marchGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(minimum, minimumGlobal, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            U initialGlobal, finalGlobal;
            MPI_Allreduce(initial.data(), initialGlobal.data(), 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(final.data(), finalGlobal.data(), 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            long long localCounts[2]{_fallbackCount, _faceCalls}, globalCounts[2]{};
            MPI_Allreduce(localCounts, globalCounts, 2, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            DNDS_check_throw_info(std::abs(vGlobal / domain.prod() - 1) < 1e-10, "Reference total box volume mismatch");
            if (_rank == 0)
            {
                nlohmann::json result = {{"problem", problem}, {"nodes", _x.size()}, {"edges", _edges.size()},
                    {"ranks", _ranks}, {"partition", "contiguous CGNS node IDs; in-place local scans"},
                    {"dt", dt}, {"end_time", end}, {"steps", steps}, {"sweeps", sweeps},
                    {"cache", "zero initially; retained across all RK stages and physical steps"},
                    {"variational_weight", 1}, {"tet_points", 8}, {"triangle_points", 6},
                    {"total_volume", vGlobal}, {"density_L1", global[0] / vGlobal},
                    {"density_L2", std::sqrt(global[1] / vGlobal)}, {"density_Linf", checksGlobal[5]},
                    {"density_L2_higher_reference", std::sqrt(global[2] / vGlobal)},
                    {"reference_quadrature_difference_L2", std::sqrt(global[3] / vGlobal)},
                    {"reference_quadrature_difference_Linf", checksGlobal[6]},
                    {"maximum_volume_mismatch", checksGlobal[0]}, {"maximum_relative_closure", checksGlobal[1]},
                    {"maximum_moment_mismatch", checksGlobal[2]}, {"maximum_matrix_condition", checksGlobal[3]},
                    {"maximum_matrix_solve_error", checksGlobal[4]}, {"first_coupled_residual", _firstReconstructionResidual},
                    {"last_coupled_residual", lastStageResidual}, {"mean_fallback_count", globalCounts[0]},
                    {"last_rhs_riemann_calls", globalCounts[1]}, {"rhs_evaluations", _rhsCalls},
                    {"minimum_mean_density", minimumGlobal[0]}, {"minimum_mean_pressure", minimumGlobal[1]},
                    {"initial_integrals", Array(initialGlobal)}, {"final_integrals", Array(finalGlobal)},
                    {"mass_drift", finalGlobal(0) - initialGlobal(0)}, {"march_seconds", marchGlobal}};
                if (problem == "tgv")
                {
                    for (const char *key : {"density_L1", "density_L2", "density_Linf", "density_L2_higher_reference",
                                           "reference_quadrature_difference_L2", "reference_quadrature_difference_Linf"})
                        result[key] = nullptr;
                    result["error_measure"] = "nodal field self-convergence; no analytic final solution";
                    result["point_field_pattern"] = std::filesystem::path(output).filename().string() + ".rank%04d.csv";
                    result["final_point_reconstruction_residual"] = finalPointResidual;
                    result["mach"] = 0.1;
                    result["mpi_ranks"] = _ranks;
                }
                if (problem == "tgv2d")
                {
                    for (const char *key : {"density_L1", "density_L2", "density_Linf", "density_L2_higher_reference",
                                           "reference_quadrature_difference_L2", "reference_quadrature_difference_Linf"})
                        result[key] = nullptr;
                    result["error_measure"] = "external 2D spectral reference projected to conservative control-volume means";
                    result["domain"] = Array(domain);
                    result["mach"] = 0.3;
                    result["dimensions"] = 2;
                    result["conservative_mean_fields"] = true;
                }
                if (problem == "entropy2d")
                {
                    result["error_measure"] = "analytic entropy field projected to conservative control-volume means";
                    result["domain"] = Array(domain);
                    result["dimensions"] = 2;
                    result["velocity"] = {1, 0.5, 0};
                    result["amplitude"] = 0.2;
                    result["conservative_mean_fields"] = true;
                }
                if (_entropyMomentCorrection) result["entropy_moment_correction"] = true;
                result["surface_flux_mode"] = _quadratureFlux ? "surface6" : "midpoint";
                if (_quadratureFlux) result["quadrature_cached"] = !_quadratureFaces.empty();
                DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite reference result");
                std::ofstream file(output); file << std::setw(2) << result << '\n';
                std::ofstream detail(output + ".audit.json"); detail << std::setw(2) << _audit << '\n';
                DNDS_check_throw_info(file.good() && detail.good(), "Reference output write failed");
                std::cout << result.dump(2) << std::endl;
            }
        }
    };
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    try
    {
        DNDS_check_throw_info(argc == 7, "usage: mesh.cgns output.json entropy|entropy_xy|entropy2d|vortex|tgv|tgv2d dt end_time sweeps");
        DNDS_check_throw_info(std::string(argv[3]) == "entropy" || std::string(argv[3]) == "entropy_xy" ||
            std::string(argv[3]) == "vortex" || std::string(argv[3]) == "tgv" ||
            std::string(argv[3]) == "tgv2d" || std::string(argv[3]) == "entropy2d", "Unknown problem");
        if (std::string(argv[3]) == "tgv2d" || std::string(argv[3]) == "entropy2d")
        {
            const char *depth = std::getenv("HOCFV_TGV2D_DEPTH");
            DNDS_check_throw_info(depth != nullptr, "Planar reference requires HOCFV_TGV2D_DEPTH");
            domain = V(2 * pi, 2 * pi, std::stod(depth));
            DNDS_check_throw_info(domain.allFinite() && domain.minCoeff() > 0, "Invalid tgv2d domain");
        }
        DNDS_check_throw_info(std::stoi(argv[6]) > 0, "Reconstruction sweeps must be positive");
        Reference solver(argv[1]);
        if (std::getenv("HOCFV_AUDIT_RHS")) solver.AuditEntropyRHS(argv[3], std::stoi(argv[6]), argv[2]);
        else solver.Run(argv[3], std::stod(argv[4]), std::stod(argv[5]), std::stoi(argv[6]), argv[2]);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Complete HOCFV reference failed: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
