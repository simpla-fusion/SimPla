/**
 * @file riemannian.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_RIEMANNIAN_MESH_H
#define SIMPLA_RIEMANNIAN_MESH_H

#include "PreDefine.h"

namespace simpla
{
namespace manifold
{
template<typename CS> using Riemannian= DefaultManifold<CS>;
}//namespace CoordinateSystem


}//namespace simpla

#endif //SIMPLA_RIEMANNIAN_MESH_H
