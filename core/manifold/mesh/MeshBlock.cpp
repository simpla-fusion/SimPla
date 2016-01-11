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
//    m_idx_max_ = properties()["Geometry"]["Topology"]["Dimensions"].template as<index_tuple>(index_tuple{10, 1, 1});
//    m_idx_min_ = 0;
    m_idx_local_max_ = m_idx_max_;
    m_idx_local_min_ = m_idx_min_;
    m_idx_memory_max_ = m_idx_max_;
    m_idx_memory_min_ = m_idx_min_;

    m_idx_local_min_ = m_idx_min_;
    m_idx_local_max_ = m_idx_max_;
    m_idx_memory_min_ = m_idx_min_;
    m_idx_memory_max_ = m_idx_max_;
    update_boundary_box();

    base::Object::touch();
}

void MeshBlock::update_boundary_box()
{
    index_tuple m_min, m_max;
    index_tuple l_min, l_max;
    index_tuple c_min, c_max;
    index_tuple ghost_width;

    m_min = traits::get<0>(memory_index_box());
    m_max = traits::get<1>(memory_index_box());

    l_min = traits::get<0>(local_index_box());
    l_max = traits::get<1>(local_index_box());

    c_min = l_min + (l_min - m_min);
    c_max = l_max - (m_max - l_max);
    m_center_box_ = std::make_tuple(c_min, c_max);

    for (int i = 0; i < ndims; ++i)
    {
        index_tuple b_min, b_max;

        b_min = l_min;
        b_max = l_max;
        b_max[i] = c_min[i];
        if (b_min[i] != b_max[i]) { m_boundary_box_.push_back(std::make_tuple(b_min, b_max)); }


        b_min = l_min;
        b_max = l_max;
        b_min[i] = c_max[i];
        if (b_min[i] != b_max[i]) { m_boundary_box_.push_back(std::make_tuple(b_min, b_max)); }


        l_min[i] = c_min[i];
        l_max[i] = c_max[i];
    }

    m_min = traits::get<0>(memory_index_box());
    m_max = traits::get<1>(memory_index_box());

    l_min = traits::get<0>(local_index_box());
    l_max = traits::get<1>(local_index_box());

    for (int i = 0; i < ndims; ++i)
    {
        index_tuple g_min, g_max;


        g_min = m_min;
        g_max = m_max;
        g_min[i] = m_min[i];
        g_max[i] = l_min[i];
        if (g_min[i] != g_max[i]) { m_ghost_box_.push_back(std::make_tuple(g_min, g_max)); }
        g_min = g_min;
        g_max = g_max;

        g_min[i] = l_max[i];
        g_max[i] = m_max[i];
        if (g_min[i] != g_max[i]) { m_ghost_box_.push_back(std::make_tuple(g_min, g_max)); }
        m_min[i] = l_min[i];
        m_max[i] = l_max[i];
    }
}

}}
