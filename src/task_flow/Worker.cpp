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


Worker::Worker() : m(nullptr) { }

Worker::Worker(mesh::MeshBase const &msh) : m(&msh) { };

Worker::~Worker() { teardown(); }

void Worker::setup() { };

void Worker::teardown() { };

std::shared_ptr<Worker> Worker::clone(mesh::MeshBase const &) const
{
    UNIMPLEMENTED;
    return std::shared_ptr<Worker>(nullptr);
};

io::IOStream &Worker::load(io::IOStream &is) const
{
    UNIMPLEMENTED;
    return is;
}


io::IOStream &Worker::save(io::IOStream &os) const
{
    for (auto const &item:m_attr_)
    {
//        os.write(item.first, item.second->data_set());
    }
    return os;
}

io::IOStream &Worker::check_point(io::IOStream &os) const
{
    UNIMPLEMENTED;
    return os;
}


std::ostream &Worker::print(std::ostream &os, int indent) const
{
    auto it = m_attr_.begin();
    auto ie = m_attr_.end();

    os << std::setw(indent) << " Worker={" << std::endl;

    os << std::setw(indent + 1) << " Type=\"" << get_class_name() << "\"," << std::endl;

    os << std::setw(indent + 1) << " Attributes= { \"" << it->first << "\"";

    ++it;

    for (; it != ie; ++it) { os << " , \"" << it->first << "\""; }

    os << "}," << std::endl;

    os << std::setw(indent) << "}," << std::endl;

    return os;
}


bool Worker::view(mesh::MeshBase const &other)
{
    m = &other;
    for (auto &item:m_attr_) { }
    UNIMPLEMENTED;
    return true;
};


void Worker::view(mesh::MeshBlockId const &) { }

void Worker::update_ghost_from(mesh::MeshBase const &other) { };

bool Worker::same_as(mesh::MeshBase const &) const { return false; };

std::vector<mesh::box_type> Worker::refine_boxes() const { return std::vector<mesh::box_type>(); }

void Worker::refine(mesh::MeshBase const &other) { };

bool Worker::coarsen(mesh::MeshBase const &other) { return false; };


void Worker::next_step(Real dt)
{
    m_time_ += dt;
    ++m_step_count_;
}
}}//namespace simpla { namespace task_flow
