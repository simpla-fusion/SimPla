//
// Created by salmon on 16-10-10.
//

#include "MeshBase.h"
#include "../toolbox/nTupleExt.h"
#include "../toolbox/PrettyStream.h"

namespace simpla { namespace mesh
{


MeshBase::MeshBase() {}

MeshBase::MeshBase(MeshBase const &other) :
        processer_id_(other.processer_id_),
        m_index_space_id_(other.m_index_space_id_),
        m_level_(other.m_level_),
        m_is_deployed_(false),
        m_block_count_(other.m_block_count_),
        m_l_dimensions_(other.m_l_dimensions_),
        m_l_start_(other.m_l_start_),
        m_g_dimensions_(other.m_g_dimensions_),
        m_g_start_(other.m_g_start_),
        m_g_min_(other.m_g_min_) {};

MeshBase::MeshBase(MeshBase &&other) :
        processer_id_(other.processer_id_),
        m_index_space_id_(other.m_index_space_id_),
        m_level_(other.m_level_),
        m_is_deployed_(other.m_is_deployed_),
        m_block_count_(other.m_block_count_),
        m_l_dimensions_(other.m_l_dimensions_),
        m_l_start_(other.m_l_start_),
        m_g_dimensions_(other.m_g_dimensions_),
        m_g_start_(other.m_g_start_),
        m_g_min_(other.m_g_min_) {};

MeshBase::~MeshBase() {}


void MeshBase::swap(MeshBase &other)
{
    std::swap(processer_id_, other.processer_id_);
    std::swap(m_index_space_id_, other.m_index_space_id_);
    std::swap(m_level_, other.m_level_);
    std::swap(m_is_deployed_, other.m_is_deployed_);

    std::swap(m_block_count_, other.m_block_count_);
    std::swap(m_l_dimensions_, other.m_l_dimensions_);
    std::swap(m_l_start_, other.m_l_start_);
    std::swap(m_g_dimensions_, other.m_g_dimensions_);
    std::swap(m_g_start_, other.m_g_start_);
    std::swap(m_g_min_, other.m_g_min_);

}

bool MeshBase::intersection(const box_type &other)
{
    return intersection(std::make_tuple(point_to_index(std::get<0>(other)), point_to_index(std::get<0>(other))));
}


bool MeshBase::intersection(const index_box_type &other)
{
    assert(!m_is_deployed_);
    for (int i = 0; i < ndims; ++i)
    {
        index_type l_lower = m_g_start_[i];
        index_type l_upper = m_g_start_[i] + m_block_count_[i];
        index_type r_lower = std::get<0>(other)[i];
        index_type r_upper = std::get<1>(other)[i];
        l_lower = std::max(l_lower, r_lower);
        l_upper = std::min(l_upper, l_upper);

        m_g_start_[i] = l_lower;
        m_block_count_[i] = static_cast<size_type>((l_upper > l_lower) ? (l_upper - l_lower) : 0);
    }

    return (m_l_dimensions_[0] * m_l_dimensions_[1] * m_l_dimensions_[2]) > 0;
};

bool MeshBase::intersection_outer(const index_box_type &other)
{
    assert(!m_is_deployed_);
    for (int i = 0; i < ndims; ++i)
    {
        index_type l_lower = m_g_start_[i] - m_l_start_[i];
        index_type l_upper = m_g_start_[i] + m_l_dimensions_[i];
        index_type r_lower = std::get<0>(other)[i];
        index_type r_upper = std::get<1>(other)[i];
        l_lower = std::max(l_lower, r_lower);
        l_upper = std::min(l_upper, l_upper);
        m_g_start_[i] = l_lower;
        m_block_count_[i] = static_cast<size_type>((l_upper > l_lower) ? (l_upper - l_lower) : 0);
    }

    return (m_l_dimensions_[0] * m_l_dimensions_[1] * m_l_dimensions_[2]) > 0;

}


void MeshBase::refine(int ratio)
{
    assert(!m_is_deployed_);
    ++m_level_;
    for (int i = 0; i < ndims; ++i)
    {
        m_block_count_[i] <<= ratio;
        m_g_dimensions_[i] <<= ratio;
        m_g_start_[i] <<= ratio;
        m_l_dimensions_[i] = 0;
        m_l_start_[i] = 0;
    }
}

void MeshBase::coarsen(int ratio)
{
    assert(!m_is_deployed_);
    --m_level_;
    for (int i = 0; i < ndims; ++i)
    {
        int mask = (1 << ratio) - 1;
        assert(m_block_count_[i] & mask == 0);
        assert(m_g_dimensions_[i] & mask == 0);
        assert(m_g_start_[i] & mask == 0);

        m_block_count_[i] >>= ratio;
        m_g_dimensions_[i] >>= ratio;
        m_g_start_[i] >>= ratio;
        m_l_dimensions_[i] = 0;
        m_l_start_[i] = 0;
    }
}

void MeshBase::deploy()
{
    if (m_is_deployed_) { return; }

    m_is_deployed_ = true;


    for (int i = 0; i < ndims; ++i)
    {
        if (m_block_count_[i] == 1)
        {
            m_ghost_width_[i] = 0;
            m_l_dimensions_[i] = 1;
            m_g_dimensions_[i] = 1;
            m_l_start_[i] = 0;
            m_g_start_[i] = 0;


        }
        if (m_l_start_[i] < m_ghost_width_[i] ||
            m_l_dimensions_[i] < m_ghost_width_[i] * 2 + m_block_count_[i])
        {
            m_l_start_[i] = m_ghost_width_[i];
            m_l_dimensions_[i] = m_ghost_width_[i] * 2 + m_block_count_[i];
        }


        assert (m_block_count_[i] > 0 ||
                m_l_dimensions_[i] >= m_block_count_[i] + m_l_start_[i] ||
                m_g_dimensions_[i] >= m_block_count_[i] + m_g_start_[i]);
    }
    m_g_min_ = m_g_start_ - m_l_start_;

}

//
//void MeshBase::foreach(std::function<void(index_type, index_type, index_type)> const &fun) const
//{
//
//#pragma omp parallel for
//    for (index_type i = 0; i < m_block_count_[0]; ++i)
//        for (index_type j = 0; j < m_block_count_[1]; ++j)
//            for (index_type k = 0; k < m_block_count_[2]; ++k)
//            {
//                fun(m_l_start_[0] + i, m_l_start_[1] + j, m_l_start_[2] + k);
//            }
//
//
//}
//
//void MeshBase::foreach(std::function<void(index_type)> const &fun) const
//{
//#pragma omp parallel for
//    for (index_type i = 0; i < m_block_count_[0]; ++i)
//        for (index_type j = 0; j < m_block_count_[1]; ++j)
//            for (index_type k = 0; k < m_block_count_[2]; ++k)
//            {
//                fun(hash(m_l_start_[0] + i, m_l_start_[1] + j, m_l_start_[2] + k));
//            }
//}
//
//void MeshBase::foreach(int iform, std::function<void(MeshEntityId const &)> const &fun) const
//{
//    int n = (iform == VERTEX || iform == VOLUME) ? 1 : 3;
//#pragma omp parallel for
//    for (index_type i = 0; i < m_block_count_[0]; ++i)
//        for (index_type j = 0; j < m_block_count_[1]; ++j)
//            for (index_type k = 0; k < m_block_count_[2]; ++k)
//                for (int l = 0; l < n; ++l)
//                {
//                    fun(pack(m_l_start_[0] + i, m_l_start_[1] + j, m_l_start_[2] + k, l));
//                }
//}


std::tuple<toolbox::DataSpace, toolbox::DataSpace>
MeshBase::data_space(MeshEntityType const &t, MeshZoneTag status) const
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
            f_start = m_g_start_;
            f_count = m_block_count_;

            m_dims = m_l_dimensions_;
            m_start = m_l_start_;
            m_count = m_block_count_;
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


EntityRange
MeshBase::range(MeshEntityType entityType, index_box_type const &b) const
{
    EntityRange res;

    res.append(MeshEntityIdCoder::make_range(std::get<0>(b), std::get<1>(b), entityType));
    return res;
}


EntityRange
MeshBase::range(MeshEntityType entityType, MeshZoneTag status) const
{
    EntityRange res;

    /**
     *   |<-----------------------------     valid   --------------------------------->|
     *   |<- not owned  ->|<-------------------       owned     ---------------------->|
     *   |----------------*----------------*---*---------------------------------------|
     *   |<---- ghost --->|                |   |                                       |
     *   |<------------ shared  ---------->|<--+--------  not shared  ---------------->|
     *   |<------------- DMZ    -------------->|<----------   not DMZ   -------------->|
     *
     */

    index_tuple m_outer_lower_, m_outer_upper_, m_inner_lower_, m_inner_upper_;
    m_outer_lower_ = m_g_min_;
    m_outer_upper_ = m_outer_lower_ + m_l_dimensions_;
    m_inner_lower_ = m_g_start_;
    m_inner_upper_ = m_g_start_ + m_block_count_;
    switch (status)
    {
        case SP_ES_ALL : //all valid
            res.append(MeshEntityIdCoder::make_range(m_outer_lower_, m_outer_upper_, entityType));
            break;
        case SP_ES_OWNED:
            res.append(MeshEntityIdCoder::make_range(m_inner_lower_, m_inner_upper_, entityType));
            break;
        case SP_ES_NON_LOCAL : // = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
        case SP_ES_SHARED : //       = 0x04,                    0b000100 shared by two or more get_mesh grid_dims
            break;
        case SP_ES_NOT_SHARED  : // = 0x08, //                       0b001000 not shared by other get_mesh grid_dims
            break;
        case SP_ES_GHOST : // = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110
            if (m_g_dimensions_[0] > 1)
            {
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_outer_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                index_tuple{m_inner_lower_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_inner_upper_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                index_tuple{m_outer_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
            }
            if (m_g_dimensions_[1] > 1)
            {
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_inner_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                index_tuple{m_inner_upper_[0], m_inner_lower_[1], m_outer_upper_[2]}, entityType));
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_inner_lower_[0], m_inner_upper_[1], m_outer_lower_[2]},
                                index_tuple{m_inner_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
            }
            if (m_g_dimensions_[2] > 1)
            {
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_inner_lower_[0], m_inner_lower_[1], m_outer_lower_[2]},
                                index_tuple{m_inner_upper_[0], m_inner_upper_[1], m_inner_lower_[2]}, entityType));
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_inner_lower_[0], m_inner_lower_[1], m_inner_upper_[2]},
                                index_tuple{m_inner_upper_[0], m_inner_upper_[1], m_outer_upper_[2]}, entityType));
            }
            break;
        case SP_ES_DMZ: //  = 0x100,
        case SP_ES_NOT_DMZ: //  = 0x200,
        case SP_ES_LOCAL : // = SP_ES_NOT_SHARED | SP_ES_OWNED, //              0b001001
            res.append(MeshEntityIdCoder::make_range(m_inner_lower_, m_inner_upper_, entityType));
            break;
        case SP_ES_VALID:
            index_tuple l, u;
            l = m_outer_lower_;
            u = m_outer_upper_;
            for (int i = 0; i < 3; ++i)
            {
                if (m_g_dimensions_[i] > 1 && m_ghost_width_[i] != 0)
                {
                    l[i] += 1;
                    u[i] -= 1;
                }
            }
            res.append(MeshEntityIdCoder::make_range(l, u, entityType));
            break;

//        case SP_ES_INTERFACE: //  = 0x010, //                        0b010000 interface(boundary) shared by two get_mesh grid_dims,
//            res.append(m_interface_entities_[entityType]);
            break;
        default:
            UNIMPLEMENTED;
            break;
    }
    return std::move(res);
};


}}