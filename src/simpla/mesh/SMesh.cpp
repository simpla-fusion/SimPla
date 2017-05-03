//
// Created by salmon on 17-4-25.
//

#include "SMesh.h"
#include <simpla/utilities/EntityIdCoder.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {

void SMesh::InitializeData(Real time_now) {
    StructuredMesh::InitializeData(time_now);
    m_vertices_.Clear();
    m_volume_.Clear();
    m_dual_volume_.Clear();
    m_inv_volume_.Clear();
    m_inv_dual_volume_.Clear();
};
}
}