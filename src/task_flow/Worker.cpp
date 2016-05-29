/** 
 * @file MeshWalker.cpp
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#include "Worker.h"
#include "../mesh/MeshAtlas.h"
#include "../mesh/MeshAttribute.h"

#include "../io/IOStream.h"

namespace simpla { namespace task_flow
{


Worker::Worker() { }

Worker::~Worker() { teardown(); }


io::IOStream &Worker::check_point(io::IOStream &os) const
{
    UNIMPLEMENTED;
    return os;
}


io::IOStream &Worker::save(io::IOStream &os) const
{
    for (auto const &item:m_attr_)
    {
//        os.write(item.first, item.second->get_dataset());
    }
    return os;
}


io::IOStream &Worker::load(io::IOStream &is) const
{
    UNIMPLEMENTED;
    return is;
}

}}//namespace simpla { namespace task_flow
