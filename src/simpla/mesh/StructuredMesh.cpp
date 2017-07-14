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

point_type StructuredMesh::GetCellWidth() const { return GetChart()->GetCellWidth(GetBlock().GetLevel()); }
point_type StructuredMesh::GetOrigin() const { return GetChart()->GetOrigin(); }
point_type StructuredMesh::map(point_type const& p) const { return GetChart()->map(p); }
point_type StructuredMesh::inv_map(point_type const& p) const { return GetChart()->inv_map(p); }
size_tuple StructuredMesh::GetDimensions() const { return GetBlock().GetDimensions(); }
index_tuple StructuredMesh::GetIndexOrigin() const { return GetBlock().GetIndexOrigin(); }
index_tuple StructuredMesh::GetGhostWidth(int tag) const { return GetBlock().GetGhostWidth(); }
box_type StructuredMesh::GetBox() const {
    box_type res;
    //    index_tuple lo, hi;
    //    std::tie(lo, hi) = GetIndexBox(VERTEX);
    //    std::get<0>(res) = point(
    //        EntityId{static_cast<int16_t>(lo[0]), static_cast<int16_t>(lo[1]), static_cast<int16_t>(lo[2]), 0},
    //        nullptr);
    //    std::get<1>(res) = point(
    //        EntityId{static_cast<int16_t>(hi[0]), static_cast<int16_t>(hi[1]), static_cast<int16_t>(hi[2]), 0},
    //        nullptr);
    return res;
}
index_box_type StructuredMesh::GetIndexBox(int tag) const {
    index_box_type res = GetBlock().GetIndexBox();
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
