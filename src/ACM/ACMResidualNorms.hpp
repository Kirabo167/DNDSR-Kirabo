#pragma once

#include "ACMTime.hpp"

namespace DNDS::ACM
{
    /// CPU field rows contain four flow components followed by the active turbulence variables.
    /// Reduce owned cells once per report. Empty ranks contribute zero and still take part.
    template <int Components, class Field>
    TimeStepReport MeasureResidualNorms(const Field &residual, index ownedCells, index globalCells, MPI_Comm comm)
    {
        static_assert(Components >= 4 && Components <= 6);
        std::array<real, Components> squares{};
        for (index cell = 0; cell < ownedCells; ++cell)
        {
            const auto &row = residual[cell];
            for (int variable = 0; variable < Components; ++variable)
                squares[variable] += row(variable) * row(variable);
        }
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(squares, comm);

        TimeStepReport report;
        const real count = std::max<real>(1, globalCells);
        real flowSquare = 0, turbulenceSquare = 0;
        for (int variable = 0; variable < Components; ++variable)
        {
            report.equationDefectNorms[variable] = std::sqrt(squares[variable] / count);
            (variable < 4 ? flowSquare : turbulenceSquare) += squares[variable];
        }
        report.finalDefectNorm = std::sqrt(flowSquare / (4 * count));
        if constexpr (Components > 4)
            report.turbulenceDefectNorm = std::sqrt(turbulenceSquare / ((Components - 4) * count));
        return report;
    }
}
