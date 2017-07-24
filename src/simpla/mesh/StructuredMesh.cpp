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

index_box_type StructuredMesh::IndexBox(int tag) const {
    index_box_type res = GetBlock()->IndexBox();
    switch (tag) {
        case 0:
            std::get<1>(res) += 1;
            break;
        case 1:
            std::get<1>(res)[1] += 1;
            std::get<1>(res)[2] += 1;
            break;
        case 2:
            std::get<1>(res)[0] += 1;
            std::get<1>(res)[2] += 1;
            break;
        case 4:
            std::get<1>(res)[0] += 1;
            std::get<1>(res)[1] += 1;
            break;
        case 3:
            std::get<1>(res)[2] += 1;
            break;
        case 5:
            std::get<1>(res)[1] += 1;
            break;
        case 6:
            std::get<1>(res)[0] += 1;
            break;
        case 7:
        default:
            break;
    }
    return res;
}

}  // namespace mesh{
}  // namespace simpla{
