//
// Created by salmon on 16-10-10.
//

#include "MeshBlock.h"

#include <simpla/algebra/nTuple.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/PrettyStream.h>
#include "EntityId.h"
namespace simpla {
namespace mesh {

MeshBlock::MeshBlock() : m_ndims_(0) {}

MeshBlock::MeshBlock(int ndims, index_type const* lo, index_type const* up, Real const* dx,
                     Real const* xlo)
    : Object(),
      m_ndims_(ndims),
      m_g_box_{{lo == nullptr ? 0 : lo[0], lo == nullptr ? 0 : lo[1], lo == nullptr ? 0 : lo[2]},
               {up == nullptr ? 1 : up[0], up == nullptr ? 1 : up[1], up == nullptr ? 1 : up[2]}},
      m_dx_{dx == nullptr ? 1 : dx[0], dx == nullptr ? 1 : dx[1], dx == nullptr ? 1 : dx[2]},
      m_level_(0) {
    m_global_origin_[0] = xlo == nullptr ? 0 : xlo[0] - std::get<0>(m_g_box_)[0] * m_dx_[0];
    m_global_origin_[1] = xlo == nullptr ? 0 : xlo[1] - std::get<0>(m_g_box_)[1] * m_dx_[1];
    m_global_origin_[2] = xlo == nullptr ? 0 : xlo[2] - std::get<0>(m_g_box_)[2] * m_dx_[2];

    deploy();
}

MeshBlock::~MeshBlock() {}

std::ostream& MeshBlock::print(std::ostream& os, int indent) const {
    os << std::setw(indent + 1) << "type = \"" << get_class_name() << "\" ,"
       << " level = " << level() << ",  box = " << m_g_box_;
    return os;
}

void MeshBlock::deploy() {
    if (concept::LifeControllable::is_deployed()) { return; }

    concept::LifeControllable::deploy();

    ASSERT(m_ndims_ <= 3);

    ASSERT(toolbox::is_valid(m_g_box_));
    for (int i = 0; i < m_ndims_; ++i) {
        if (std::get<1>(m_g_box_)[i] <= std::get<0>(m_g_box_)[i] + 1) {
            m_ghost_width_[i] = 0;

            std::get<0>(m_m_box_)[i] = 0;
            std::get<1>(m_m_box_)[i] = 1;

            std::get<0>(m_m_box_)[i] = 0;
            std::get<1>(m_m_box_)[i] = 1;

            std::get<0>(m_inner_box_)[i] = 0;
            std::get<1>(m_inner_box_)[i] = 1;

            std::get<0>(m_outer_box_)[i] = 0;
            std::get<1>(m_outer_box_)[i] = 1;

            m_inv_dx_[i] = 0;

            m_l2g_scale_[i] = 0;
            m_l2g_shift_[i] = m_global_origin_[i];

            m_g2l_scale_[i] = 0;
            m_g2l_shift_[i] = 0;

        } else {
            m_inv_dx_[i] = static_cast<Real>(1.0) / m_dx_[i];

            m_l2g_scale_[i] = m_dx_[i];
            m_l2g_shift_[i] = m_global_origin_[i];

            m_g2l_scale_[i] = m_inv_dx_[i];
            m_g2l_shift_[i] = -(m_global_origin_[i]) * m_g2l_scale_[i];
        }
    }

    m_inner_box_ = m_g_box_;
    m_outer_box_ = m_g_box_;

    std::get<0>(m_outer_box_) -= m_ghost_width_;
    std::get<1>(m_outer_box_) += m_ghost_width_;

    m_m_box_ = m_outer_box_;
}

std::shared_ptr<MeshBlock> MeshBlock::create(int inc_level, const index_type* lo,
                                             const index_type* hi) const {
    auto res = std::make_shared<MeshBlock>();
    if (inc_level >= 0) {
        std::get<0>(res->m_g_box_)[0] = lo[0] << inc_level;
        std::get<0>(res->m_g_box_)[1] = lo[1] << inc_level;
        std::get<0>(res->m_g_box_)[2] = lo[2] << inc_level;
        std::get<1>(res->m_g_box_)[0] = hi[0] << inc_level;
        std::get<1>(res->m_g_box_)[1] = hi[1] << inc_level;
        std::get<1>(res->m_g_box_)[2] = hi[2] << inc_level;
    } else if (inc_level < 0) {
        std::get<0>(res->m_g_box_)[0] = lo[0] >> -inc_level;
        std::get<0>(res->m_g_box_)[1] = lo[1] >> -inc_level;
        std::get<0>(res->m_g_box_)[2] = lo[2] >> -inc_level;
        std::get<1>(res->m_g_box_)[0] = hi[0] >> -inc_level;
        std::get<1>(res->m_g_box_)[1] = hi[1] >> -inc_level;
        std::get<1>(res->m_g_box_)[2] = hi[2] >> -inc_level;
    }
    res->m_level_ += inc_level;
    res->deploy();
    return res;
}

std::shared_ptr<MeshBlock> MeshBlock::intersection(index_box_type const& other_box, int inc_level) {
    return create(inc_level, toolbox::intersection(m_inner_box_, other_box));
}

Range<MeshEntityId> MeshBlock::range(box_type const& b, size_type entityType, size_type dof) const {
    index_tuple l, u;
    l = point_to_index(std::get<1>(b));
    u = point_to_index(std::get<1>(b)) + 1;
    return std::move(range(&l[0], &u[0], entityType, dof));
}

Range<MeshEntityId> MeshBlock::range(index_box_type const& b, size_type iform,
                                     size_type dof) const {
    return std::move(range(&(std::get<0>(b)[0]), &(std::get<1>(b)[0]), iform, dof));
}

Range<MeshEntityId> MeshBlock::range(index_type const* b, index_type const* e, size_type entityType,
                                     size_type dof) const {
    return std::move(
        Range<MeshEntityId>(std::make_shared<ContinueRange<MeshEntityId>>(b, e, entityType, dof)));
    //    return std::move(make_continue_range<MeshEntityId>(b, e, entityType, dof));
}

Range<MeshEntityId> MeshBlock::range(MeshZoneTag status, size_type entityType,
                                     size_type dof) const {
    Range<mesh::MeshEntityId> res;

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
    std::tie(m_outer_lower_, m_outer_upper_) = m_outer_box_;
    std::tie(m_inner_lower_, m_inner_upper_) = m_inner_box_;
    size_tuple m_g_dimensions_;
    m_g_dimensions_ = toolbox::dimensions(m_g_box_);
    switch (status) {
        case SP_ES_ALL:  // all valid
            res.append(std::make_shared<ContinueRange<MeshEntityId>>(m_outer_box_, entityType, dof));
            break;
        case SP_ES_OWNED:
            res.append(std::make_shared<ContinueRange<MeshEntityId>>(m_inner_box_, entityType, dof));
            break;
        case SP_ES_NON_LOCAL:  // = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
        case SP_ES_SHARED:     //       = 0x04,                    0b000100 shared by two or more
                               //       get_mesh grid_dims
            break;
        case SP_ES_NOT_SHARED:  // = 0x08, //                       0b001000 not shared by other
                                // get_mesh grid_dims
            break;
        case SP_ES_GHOST:  // = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110
            if (m_g_dimensions_[0] > 1) {
                res.append(std::make_shared<ContinueRange<MeshEntityId>>(
                    index_box_type{{m_outer_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                   {m_inner_lower_[0], m_outer_upper_[1], m_outer_upper_[2]}},
                    entityType, dof));
                res.append(std::make_shared<ContinueRange<MeshEntityId>>(
                    index_box_type{{m_inner_upper_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                   {m_outer_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}},
                    entityType, dof));
            }
            if (m_g_dimensions_[1] > 1) {
                res.append(std::make_shared<ContinueRange<MeshEntityId>>(
                    index_box_type{{m_inner_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                   {m_inner_upper_[0], m_inner_lower_[1], m_outer_upper_[2]}},
                    entityType, dof));
                res.append(std::make_shared<ContinueRange<MeshEntityId>>(
                    index_box_type{{m_inner_lower_[0], m_inner_upper_[1], m_outer_lower_[2]},
                                   {m_inner_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}},
                    entityType, dof));
            }
            if (m_g_dimensions_[2] > 1) {
                res.append(std::make_shared<ContinueRange<MeshEntityId>>(
                    index_box_type{{m_inner_lower_[0], m_inner_lower_[1], m_outer_lower_[2]},
                                   {m_inner_upper_[0], m_inner_upper_[1], m_inner_lower_[2]}},
                    entityType, dof));
                res.append(std::make_shared<ContinueRange<MeshEntityId>>(
                    index_box_type{{m_inner_lower_[0], m_inner_lower_[1], m_inner_upper_[2]},
                                   {m_inner_upper_[0], m_inner_upper_[1], m_outer_upper_[2]}},
                    entityType, dof));
            }
            break;
        case SP_ES_DMZ:      //  = 0x100,
        case SP_ES_NOT_DMZ:  //  = 0x200,
        case SP_ES_LOCAL:    // = SP_ES_NOT_SHARED | SP_ES_OWNED, //              0b001001
            res.append(
                std::make_shared<ContinueRange<MeshEntityId>>(m_inner_box_, entityType, dof));
            break;
        case SP_ES_VALID: {
            index_tuple l, u;
            l = m_outer_lower_;
            u = m_outer_upper_;
            for (int i = 0; i < 3; ++i) {
                if (m_g_dimensions_[i] > 1 && m_ghost_width_[i] != 0) {
                    l[i] += 1;
                    u[i] -= 1;
                }
            }
            res.append(std::make_shared<ContinueRange<MeshEntityId>>(l, u, entityType, dof));
            break;
        }
        //        case SP_ES_INTERFACE: //  = 0x010, //                        0b010000
        //        interface(boundary) shared by two get_mesh grid_dims,
        //            res.append(m_interface_entities_[entityType]);
        break;
        default:
            UNIMPLEMENTED;
            break;
    }
    return std::move(res);
};
}
}