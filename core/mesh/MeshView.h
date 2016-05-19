/**
 * @file MeshView.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESH_MESHVIEW_H
#define SIMPLA_MESH_MESHVIEW_H

#include "Mesh.h"

namespace simpla { namespace mesh
{

class ViewBase
{
    virtual EntityRange range();
};

template<typename ...>
class View : public ViewBase
{

};

}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESH_MESHVIEW_H
