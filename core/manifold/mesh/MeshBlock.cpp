/**
 * @file MeshBlock.cpp
 * @author salmon
 * @date 2015-12-23.
 */
#include "MeshBlock.h"

namespace simpla { namespace mesh
{


MeshBlock::MeshBlock()
{
    m_idx_min_ = 0;
    m_idx_max_ = 0;
    m_idx_local_min_ = m_idx_min_;
    m_idx_local_max_ = m_idx_max_;
    m_idx_memory_min_ = m_idx_min_;
    m_idx_memory_max_ = m_idx_max_;
}


MeshBlock::~MeshBlock() { }

//            base_type(other),
//            m_idx_min_(other.m_idx_min_),
//            m_idx_max_(other.m_idx_max_),
//            m_idx_local_min_(other.m_idx_local_min_),
//            m_idx_local_max_(other.m_idx_local_max_),
//            m_idx_memory_min_(other.m_idx_memory_min_),
//            m_idx_memory_max_(other.m_idx_memory_max_)
//    {
//
//    }


//    virtual void swap(this_type &other)
//    {
//        base_type::swap(other);
//
//        std::swap(m_idx_min_, other.m_idx_min_);
//        std::swap(m_idx_max_, other.m_idx_max_);
//        std::swap(m_idx_local_min_, other.m_idx_local_min_);
//        std::swap(m_idx_local_max_, other.m_idx_local_max_);
//        std::swap(m_idx_memory_min_, other.m_idx_memory_min_);
//        std::swap(m_idx_memory_max_, other.m_idx_memory_max_);
//        deploy();
//        other.deploy();
//
//    }






typename MeshBlock::index_tuple MeshBlock::dimensions() const
{
    index_tuple res;

    res = m_idx_max_ - m_idx_min_;

    return std::move(res);
}

void MeshBlock::box(box_type const &b)
{
    m_x_min_ = b[0];
    m_x_max_ = b[1];
    properties()["Box"] = b;
    touch();
};

typename MeshBlock::box_type MeshBlock::box() const
{
    return traits::make_nTuple(m::point(m_idx_min_), m::point(m_idx_max_));
};

typename MeshBlock::box_type MeshBlock::box(id_type const &s) const
{
    return traits::make_nTuple(m::point(s - _DA), m::point(s + _DA));
};

typename MeshBlock::box_type MeshBlock::local_box() const
{
    return traits::make_nTuple(m::point(m_idx_local_min_), m::point(m_idx_local_max_));
};


void MeshBlock::decompose(index_tuple const &dist_dimensions, index_tuple const &dist_coord,
                          index_type gw)
{


    index_tuple b, e;
    b = m_idx_local_min_;
    e = m_idx_local_max_;
    for (int n = 0; n < ndims; ++n)
    {

        m_idx_local_min_[n] = b[n] + (e[n] - b[n]) * dist_coord[n] / dist_dimensions[n];

        m_idx_local_max_[n] = b[n] + (e[n] - b[n]) * (dist_coord[n] + 1) / dist_dimensions[n];

        if (dist_dimensions[n] > 1)
        {
            if (m_idx_local_max_[n] - m_idx_local_min_[n] >= 2 * gw)
            {
                m_idx_memory_min_[n] = m_idx_local_min_[n] - gw;
                m_idx_memory_max_[n] = m_idx_local_max_[n] + gw;
            }
            else
            {
                VERBOSE << "mesh block decompose failed! Block dimension is smaller than process grid. "
                << m_idx_local_min_ << m_idx_local_max_
                << dist_dimensions << dist_coord << std::endl;
                THROW_EXCEPTION_RUNTIME_ERROR(
                        "mesh block decompose failed! Block dimension is smaller than process grid. ");
            }
        }
    }
    update_boundary_box();

}

void MeshBlock::deploy()
{
    m_idx_max_ = properties()["Geometry"]["Topology"]["Dimensions"].template as<index_tuple>(index_tuple{10, 1, 1});
    m_idx_min_ = 0;
    m_idx_local_max_ = m_idx_max_;
    m_idx_local_min_ = m_idx_min_;
    m_idx_memory_max_ = m_idx_max_;
    m_idx_memory_min_ = m_idx_min_;

    m_is_valid_ = true;
    m_idx_local_min_ = m_idx_min_;
    m_idx_local_max_ = m_idx_max_;
    m_idx_memory_min_ = m_idx_min_;
    m_idx_memory_max_ = m_idx_max_;
    update_boundary_box();
}

void MeshBlock::update_boundary_box()
{
    nTuple<size_t, ndims> m_min, m_max;
    nTuple<size_t, ndims> l_min, l_max;
    nTuple<size_t, ndims> c_min, c_max;
    nTuple<size_t, ndims> ghost_width;

    m_min = traits::get<0>(memory_index_box());
    m_max = traits::get<1>(memory_index_box());

    l_min = traits::get<0>(local_index_box());
    l_max = traits::get<1>(local_index_box());

    c_min = l_min + (l_min - m_min);
    c_max = l_max - (m_max - l_max);
    m_center_box_ = traits::make_nTuple(c_min, c_max);

    for (int i = 0; i < ndims; ++i)
    {
        nTuple<size_t, ndims> b_min, b_max;

        b_min = l_min;
        b_max = l_max;
        b_max[i] = c_min[i];
        if (b_min[i] != b_max[i]) { m_boundary_box_.push_back(traits::make_nTuple(b_min, b_max)); }


        b_min = l_min;
        b_max = l_max;
        b_min[i] = c_max[i];
        if (b_min[i] != b_max[i]) { m_boundary_box_.push_back(traits::make_nTuple(b_min, b_max)); }


        l_min[i] = c_min[i];
        l_max[i] = c_max[i];
    }

    m_min = traits::get<0>(memory_index_box());
    m_max = traits::get<1>(memory_index_box());

    l_min = traits::get<0>(local_index_box());
    l_max = traits::get<1>(local_index_box());

    for (int i = 0; i < ndims; ++i)
    {
        nTuple<size_t, ndims> g_min, g_max;


        g_min = m_min;
        g_max = m_max;
        g_min[i] = m_min[i];
        g_max[i] = l_min[i];
        if (g_min[i] != g_max[i]) { m_ghost_box_.push_back(traits::make_nTuple(g_min, g_max)); }
        g_min = g_min;
        g_max = g_max;

        g_min[i] = l_max[i];
        g_max[i] = m_max[i];
        if (g_min[i] != g_max[i]) { m_ghost_box_.push_back(traits::make_nTuple(g_min, g_max)); }
        m_min[i] = l_min[i];
        m_max[i] = l_max[i];
    }
}

}}
