//
// Created by salmon on 16-10-10.
//

#include "Block.h"

namespace simpla { namespace mesh
{


Block::Block() {}

Block::Block(Block const &other) :
        processer_id_(other.processer_id_),
        m_index_space_id_(other.m_index_space_id_),
        m_level_(other.m_level_),
        m_is_deployed_(false),
        m_b_dimensions_(other.m_b_dimensions_),
        m_l_dimensions_(other.m_l_dimensions_),
        m_l_offset_(other.m_l_offset_),
        m_g_dimensions_(other.m_g_dimensions_),
        m_g_offset_(other.m_g_offset_) {};

Block::Block(Block &&other) :
        processer_id_(other.processer_id_),
        m_index_space_id_(other.m_index_space_id_),
        m_level_(other.m_level_),
        m_is_deployed_(other.m_is_deployed_),
        m_b_dimensions_(other.m_b_dimensions_),
        m_l_dimensions_(other.m_l_dimensions_),
        m_l_offset_(other.m_l_offset_),
        m_g_dimensions_(other.m_g_dimensions_),
        m_g_offset_(other.m_g_offset_) {};

Block::~Block() {}


void Block::swap(Block &other)
{
    std::swap(processer_id_, other.processer_id_);
    std::swap(m_index_space_id_, other.m_index_space_id_);
    std::swap(m_level_, other.m_level_);
    std::swap(m_is_deployed_, other.m_is_deployed_);

//    std::swap(m_b_dimensions_, other.m_b_dimensions_);
//    std::swap(m_l_dimensions_, other.m_l_dimensions_);
//    std::swap(m_l_offset_, other.m_l_offset_);
//    std::swap(m_g_dimensions_, other.m_g_dimensions_);
//    std::swap(m_g_offset_, other.m_g_offset_);
}


void Block::intersection(const index_box_type &other)
{
    assert(!m_is_deployed_);
    for (int i = 0; i < ndims; ++i)
    {
        index_type l_lower = m_g_offset_[i];
        index_type l_upper = m_g_offset_[i] + m_b_dimensions_[i];
        index_type r_lower = std::get<0>(other)[i];
        index_type r_upper = std::get<1>(other)[i];
        l_lower = std::max(l_lower, r_lower);
        l_upper = std::min(l_upper, l_upper);
        m_b_dimensions_[i] = (l_upper > l_lower) ? (l_upper - l_lower) : 0;
        m_g_offset_[i] = l_lower;
    }
};

void Block::refine(int ratio)
{
    assert(!m_is_deployed_);
    ++m_level_;
    for (int i = 0; i < ndims; ++i)
    {
        m_b_dimensions_[i] <<= ratio;
        m_g_dimensions_[i] <<= ratio;
        m_g_offset_[i] <<= ratio;
        m_l_dimensions_[i] = 0;
        m_l_offset_[i] = 0;
    }
}

void Block::coarsen(int ratio)
{
    assert(!m_is_deployed_);
    --m_level_;
    for (int i = 0; i < ndims; ++i)
    {
        int mask = (1 << ratio) - 1;
        assert(m_b_dimensions_[i] & mask == 0);
        assert(m_g_dimensions_[i] & mask == 0);
        assert(m_g_offset_[i] & mask == 0);

        m_b_dimensions_[i] >>= ratio;
        m_g_dimensions_[i] >>= ratio;
        m_g_offset_[i] >>= ratio;
        m_l_dimensions_[i] = 0;
        m_l_offset_[i] = 0;
    }
}

void Block::deploy()
{
    if (m_is_deployed_) { return; }

    for (int i = 0; i < ndims; ++i)
    {
        if (m_l_offset_[i] < m_ghost_width_[i] ||
            m_l_dimensions_[i] < m_ghost_width_[i] * 2 + m_b_dimensions_[i])
        {
            m_l_offset_[i] = m_ghost_width_[i];
            m_l_dimensions_[i] = m_ghost_width_[i] * 2 + m_b_dimensions_[i];
        }

        if (m_b_dimensions_[i] < 1 ||
            m_l_dimensions_[i] < m_b_dimensions_[i] + m_l_offset_[i] ||
            m_g_dimensions_[i] < m_b_dimensions_[i] + m_g_offset_[i]) { m_is_deployed_ = false; }
    }
    m_is_deployed_ = true;
}


void Block::for_each(std::function<void(index_type, index_type, index_type)> const &fun) const
{
#pragma omp parallel for
    for (index_type i = m_l_offset_[0]; i < m_l_offset_[0] + m_b_dimensions_[0]; ++i)
        for (index_type j = m_l_offset_[1]; j < m_l_offset_[1] + m_b_dimensions_[1]; ++j)
            for (index_type k = m_l_offset_[2]; k < m_l_offset_[2] + m_b_dimensions_[2]; ++k) { fun(i, j, k); }
}

void Block::for_each(std::function<void(index_type)> const &fun) const
{
#pragma omp parallel for
    for (index_type i = m_l_offset_[0]; i < m_l_offset_[0] + m_b_dimensions_[0]; ++i)
        for (index_type j = m_l_offset_[1]; j < m_l_offset_[1] + m_b_dimensions_[1]; ++j)
            for (index_type k = m_l_offset_[2]; k < m_l_offset_[2] + m_b_dimensions_[2]; ++k) { fun(hash(i, j, k)); }
}

void Block::for_each(int iform, std::function<void(MeshEntityId const &)> const &fun) const
{
    int n = (iform == VERTEX || iform == VOLUME) ? 1 : 3;
#pragma omp parallel for
    for (index_type i = m_l_offset_[0]; i < m_l_offset_[0] + m_b_dimensions_[0]; ++i)
        for (index_type j = m_l_offset_[1]; j < m_l_offset_[1] + m_b_dimensions_[1]; ++j)
            for (index_type k = m_l_offset_[2]; k < m_l_offset_[2] + m_b_dimensions_[2]; ++k)
                for (int l = 0; l < n; ++l) { fun(pack(i, j, k, l)); }
}


std::tuple<toolbox::DataSpace, toolbox::DataSpace>
Block::data_space(MeshEntityType const &t, MeshEntityStatus status) const
{
    int i_ndims = (t == EDGE || t == FACE) ? (ndims + 1) : ndims;

    nTuple<size_type, ndims + 1> f_dims, f_count;
    nTuple<size_type, ndims + 1> f_start;

    nTuple<size_type, ndims + 1> m_dims, m_count;
    nTuple<size_type, ndims + 1> m_start;

    switch (status)
    {
        case SP_ES_ALL:
            f_dims = m_l_dimensions_;//+ m_offset_;
            f_start = 0;//m_offset_;
            f_count = m_l_dimensions_;

            m_dims = m_l_dimensions_;
            m_start = 0;
            m_count = m_l_dimensions_;
            break;
        case SP_ES_OWNED:
        default:
            f_dims = m_g_dimensions_;
            f_start = m_g_offset_;
            f_count = m_b_dimensions_;

            m_dims = m_l_dimensions_;
            m_start = m_l_offset_;
            m_count = m_b_dimensions_;
            break;

    }
    f_dims[ndims] = 3;
    f_start[ndims] = 0;
    f_count[ndims] = 3;


    m_dims[ndims] = 3;
    m_start[ndims] = 0;
    m_count[ndims] = 3;

    toolbox::DataSpace f_space(i_ndims, &f_dims[0]);
    f_space.select_hyperslab(&f_start[0], nullptr, &f_count[0], nullptr);

    toolbox::DataSpace m_space(i_ndims, &m_dims[0]);
    m_space.select_hyperslab(&m_start[0], nullptr, &m_count[0], nullptr);

    return std::forward_as_tuple(m_space, f_space);

};
}}