/** 
 * @file MeshUtility.h
 * @author salmon
 * @date 16-6-2 - 上午7:16
 *  */

#ifndef SIMPLA_MESHUTILITIY_H
#define SIMPLA_MESHUTILITIY_H

#include "MeshCommon.h"
#include "MeshAtlas.h"
#include "MeshBase.h"

namespace simpla { namespace mesh
{

//std::map<index_tuple, box_type> decompose(box_type const &b, index_tuple const &d)
//{
//
//};


MeshEntityRange select(MeshBase const &m, MeshEntityRange const &r, box_type const &b);
}}//namespace simpla {namespace get_mesh
#endif //SIMPLA_MESHUTILITIY_H
