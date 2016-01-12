/**
 * @file MeshBlock.cpp
 * @author salmon
 * @date 2015-12-23.
 */
#include "MeshBlock.h"

namespace simpla { namespace mesh
{


MeshBlock::MeshBlock() { }


MeshBlock::~MeshBlock() { }

void MeshBlock::dimensions(index_tuple const &d)
{
    m_dimensions_ = d;
    m_ndims_ = 0;
    for (int n = 0; n < ndims; ++n)
    {
        if (m_dimensions_[n] <= 1)
        {
            m_dimensions_[n] = 1;

            m_idx_min_[n] = 0;

            m_idx_max_[n] = 1;

            m_idx_local_min_[n] = 0;

            m_idx_local_max_[n] = 1;

            m_idx_memory_min_[n] = 0;

            m_idx_memory_max_[n] = 1;
        }
        else
        {
            ++m_ndims_;

            m_idx_min_[n] = 0;

            m_idx_max_[n] = m_dimensions_[n] + m_idx_min_[n];

            m_idx_local_min_[n] = m_idx_min_[n];

            m_idx_local_max_[n] = m_idx_max_[n];

            m_idx_memory_min_[n] = m_idx_local_min_[n];

            m_idx_memory_max_[n] = m_idx_local_max_[n];

        }


    }
}

void MeshBlock::decompose(index_tuple const &dist_dimensions, index_tuple const &dist_coord,
                          index_type gw)
{

    m_ndims_ = 0;
    for (int n = 0; n < ndims; ++n)
    {
        index_type i_min = m_idx_local_min_[n];

        index_type i_max = m_idx_local_max_[n];

        if ((i_max - i_min) > 2 * gw * dist_dimensions[n])
        {

            m_idx_local_min_[n] = i_min + (i_max - i_min) * dist_coord[n] / dist_dimensions[n];

            m_idx_local_max_[n] = i_min + (i_max - i_min) * (dist_coord[n] + 1) / dist_dimensions[n];

            m_idx_memory_min_[n] = m_idx_local_min_[n] - gw;

            m_idx_memory_max_[n] = m_idx_local_max_[n] + gw;
        }
        else if ((i_max - i_min) > 1)
        {

            VERBOSE << "mesh block decompose failed! Block dimension is smaller than process grid. "
            << m_dimensions_ << dist_dimensions << dist_coord << std::endl;

            THROW_EXCEPTION_RUNTIME_ERROR(
                    "mesh block decompose failed! Block dimension is smaller than process grid. ");
        }


    }


}

void MeshBlock::deploy2()
{

    for (int i = 0; i < ndims; ++i)
    {
        ASSERT((m_max_[i] - m_min_[i] > EPSILON));

        m_dx_[i] = (m_max_[i] - m_min_[i]) / static_cast<Real>(m_dimensions_[i]);
    }

    point_type src_min_, src_max_;

    src_min_ = m::point(m_idx_min_);
    src_max_ = m::point(m_idx_max_);

    point_type dest_min = m_min_, dest_max = m_max_;


    for (int i = 0; i < 3; ++i)
    {
        m_map_scale_[i] = (dest_max[i] - dest_min[i]) / (src_max_[i] - src_min_[i]);

        m_inv_map_scale_[i] = (src_max_[i] - src_min_[i]) / (dest_max[i] - dest_min[i]);


        m_map_orig_[i] = dest_min[i] - src_min_[i] * m_map_scale_[i];

        m_inv_map_orig_[i] = src_min_[i] - dest_min[i] * m_inv_map_scale_[i];

    }


    index_tuple m_min, m_max;
    index_tuple l_min, l_max;
    index_tuple c_min, c_max;
    index_tuple ghost_width;

    m_min = m_idx_memory_min_;
    m_max = m_idx_memory_max_;

    l_min = m_idx_local_min_;
    l_max = m_idx_local_max_;

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

    m_min = m_idx_memory_min_;
    m_max = m_idx_memory_max_;

    l_min = m_idx_local_min_;
    l_max = m_idx_local_max_;

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

