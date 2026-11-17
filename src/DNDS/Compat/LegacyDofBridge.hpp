#pragma once

#include "DNDS/Compat/LegacyArrayBridge.hpp"
#include "DNDS/Algebra/DofField.hpp"

namespace DNDS::Runtime
{
    /// Explicit zero-copy bridge for existing real-valued owner/ghost pairs.
    /// Reuse one DofSpace for fields on the same row order. Matrix rows keep
    /// their original flat component order; no transpose or repacking occurs.
    /// Finish all managed work before legacy raw arithmetic or MPI accesses.
    /// Rebuild the binding after either legacy array replaces its allocation.
    template <class LegacyPair>
    Algebra::DofField<real> ShareLegacyDofField(Algebra::DofSpace space, LegacyPair &source)
    {
        DNDS_check_throw_info(source.father && source.son, "legacy field pair is not allocated");
        auto owned = ShareLegacyHost(*source.father);
        auto ghost = ShareLegacyHost(*source.son);
        return Algebra::DofField<real>(std::move(space), std::move(owned), std::move(ghost));
    }
}
