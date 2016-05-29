/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#include "MeshWorker.h"
#include "MeshAtlas.h"
#include "MeshAttribute.h"

#include "../io/IOStream.h"

namespace simpla { namespace mesh
{
io::IOStream &MeshWorker::check_point(io::IOStream &os) const
{
    UNIMPLEMENTED;
    return os;
}


io::IOStream &MeshWorker::save(io::IOStream &os) const
{
    for (auto const &item:m_attr_)
    {
        os.write(item.first, item.second->get_dataset());
    }
    return os;
}


io::IOStream &MeshWorker::load(io::IOStream &is) const
{
    UNIMPLEMENTED;
    return is;
}

}}//namespace simpla { namespace mesh
