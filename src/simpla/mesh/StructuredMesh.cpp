//
// Created by salmon on 17-4-25.
//
#include "StructuredMesh.h"
#include "simpla/algebra/Algebra.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/geometry/GeoAlgorithm.h"
namespace simpla {
namespace mesh {

/**
*\verbatim
*                ^s (dl)
*               /
*   (dz) t     /
*        ^    /
*        |  110-------------111
*        |  /|              /|
*        | / |             / |
*        |/  |            /  |
*       100--|----------101  |
*        | m |           |   |
*        |  010----------|--011
*        |  /            |  /
*        | /             | /
*        |/              |/
*       000-------------001---> r (dr)
*
*\endverbatim
*/

index_box_type StructuredMesh::GetIndexBox(int tag) const {
    index_tuple lo, hi;
    std::tie(lo, hi) = GetMeshBlock()->GetIndexBox();
    switch (tag) {
        case 0:
            hi += 1;
            break;
        case 1:
            hi[1] += 1;
            hi[2] += 1;
            break;
        case 2:
            hi[0] += 1;
            hi[2] += 1;
            break;
        case 4:
            hi[0] += 1;
            hi[1] += 1;
            break;
        case 3:
            hi[2] += 1;
            break;
        case 5:
            hi[1] += 1;
            break;
        case 6:
            hi[0] += 1;
            break;
        case 7:
        default:
            break;
    }
    return std::make_tuple(lo, hi);
}

}  // namespace mesh{
}  // namespace simpla{
