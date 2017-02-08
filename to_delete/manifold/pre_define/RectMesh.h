//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_MESHBOX_H
#define SIMPLA_MESHBOX_H

#include <vector>
#include <iomanip>

#include "../toolbox/macro.h"
#include "../sp_def.h"
#include "../toolbox/nTuple.h"
#include "../toolbox/nTupleExt.h"
#include "../toolbox/PrettyStream.h"
#include "../toolbox/type_traits.h"
#include "../toolbox/type_cast.h"
#include "../toolbox/Log.h"

#include "MeshCommon.h"
#include "Chart.h"
#include "EntityId.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */
struct RectMesh : public Chart, public MeshEntityIdCoder
{
private:
    typedef RectMesh this_type;
    typedef Chart base_type;
public:
    typedef Real scalar_type;

    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string getClassName() const { return class_name(); }

    static std::string class_name() { return std::string("RectMesh"); }

    /**
 *
 *   -----------------------------5
 *   |                            |
 *   |     ---------------4       |
 *   |     |              |       |
 *   |     |  ********3   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  2********   |       |
 *   |     1---------------       |
 *   0-----------------------------
 *
 *	5-0 = dimensions
 *	4-1 = e-d = ghosts
 *	2-1 = counts
 *
 *	0 = id_begin
 *	5 = id_end
 *
 *	1 = id_local_outer_begin
 *	4 = id_local_outer_end
 *
 *	2 = id_local_inner_begin
 *	3 = id_local_inner_end
 *
 *
 */

    point_type m_coords_lower_{{0, 0, 0}};

    point_type m_coords_upper_{{1, 1, 1}};

    index_tuple m_ghost_width_{{0, 0, 0}};

    index_tuple m_offset_{{0, 0, 0}};

    index_tuple m_dims_{{10, 10, 10}};


    index_tuple m_shape_{{10, 10, 10}};

    index_tuple m_lower_{{0, 0, 0}}, m_upper_{{10, 10, 10}};

    index_tuple m_inner_lower_{{0, 0, 0}}, m_inner_upper_{{10, 10, 10}};

    index_tuple m_outer_lower_{{0, 0, 0}}, m_outer_upper_{{10, 10, 10}};


    typedef MeshEntityIdCoder m;

    typedef MeshEntityId id_type;

public:
    static constexpr int ndims = 3;

    RectMesh() {}

    RectMesh(this_type const &other) : Chart(other)
    {
        m_dims_ = other.m_dims_;
        m_coords_lower_ = other.m_coords_lower_;
        m_coords_upper_ = other.m_coords_upper_;
        m_ghost_width_ = other.m_ghost_width_;
        m_offset_ = other.m_offset_;
        update();
    };

    virtual  ~RectMesh() {}

    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << std::setw(indent + 1) << " " << "Name =\"" << name() << "\"," << std::endl;
        os << std::setw(indent + 1) << " " << "Topology = { Type = \"CartesianGeometry\", "
           << "Dimensions = " << dimensions() << " m_global_start_ = " << offset() << " },"
           << std::endl;
        os << std::setw(indent + 1) << " " << "Box = " << box() << "," << std::endl;
#ifndef NDEBUG
        os
                << std::setw(indent + 1) << " " << "      lower = " << m_lower_ << "," << std::endl
                << std::setw(indent + 1) << " " << "      upper = " << m_upper_ << "," << std::endl
                << std::setw(indent + 1) << " " << "outer lower = " << m_outer_lower_ << "," << std::endl
                << std::setw(indent + 1) << " " << "outer upper = " << m_outer_upper_ << "," << std::endl
                << std::setw(indent + 1) << " " << "inner lower = " << m_inner_lower_ << "," << std::endl
                << std::setw(indent + 1) << " " << "inner upper = " << m_inner_upper_ << "," << std::endl
                << std::endl;
#endif
        return os;
    }


    virtual io::IOStream &save(io::IOStream &os) const
    {
//        os.open(type_cast<std::string>(this->short_id()) + "/");
//        os.set_attribute(".topology_dims", dimensions());
//        os.set_attribute(".box", box());
        return os;
    };


    void dimensions(index_tuple const &d) { m_dims_ = d; }

    index_tuple const &dimensions() const { return m_dims_; }

    void offset(index_tuple const &d) { m_offset_ = d; }

    virtual index_tuple offset() const { return m_offset_; }

    virtual point_type origin_point() const { return m_coords_lower_; };

    virtual void ghost_width(index_tuple const &d) { m_ghost_width_ = d; }

    virtual index_tuple const &ghost_width() const { return m_ghost_width_; }

    template<typename X0, typename X1>
    void box(X0 const &x0, X1 const &x1)
    {
        m_coords_lower_ = x0;
        m_coords_upper_ = x1;
    }

    void box(box_type const &b) { std::tie(m_coords_lower_, m_coords_upper_) = b; }

    virtual mesh::Chart &shift(index_tuple const &offset)
    {
        for (int i = 0; i < 3; ++i) { if (m_dims_[i] > 1) { m_offset_[i] += offset[i]; }}
        return *this;
    };

    virtual mesh::Chart &stretch(index_tuple const &dims)
    {
        for (int i = 0; i < 3; ++i) { if (m_dims_[i] > 1) { m_dims_[i] = dims[i]; }}
        return *this;
    };

public:

    typedef typename MeshEntityIdCoder::range_type block_range_type;


    virtual EntityRange range(box_type const &b, MeshEntityType entityType = VERTEX) const
    {
        return range(index_box(b), entityType);
    }

    virtual EntityRange range(index_box_type const &b, MeshEntityType entityType = VERTEX) const
    {
        return MeshEntityIdCoder::make_range(b, entityType);
    }

    virtual EntityRange range(MeshEntityType entityType = VERTEX, MeshEntityStatus status = SP_ES_OWNED) const
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
        switch (status)
        {
            case SP_ES_ALL : //all valid
                res.append(MeshEntityIdCoder::make_range(m_outer_lower_, m_outer_upper_, entityType));
                break;
            case SP_ES_NON_LOCAL : // = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
            case SP_ES_SHARED : //       = 0x04,                    0b000100 shared by two or more get_mesh grid_dims
                break;
            case SP_ES_NOT_SHARED  : // = 0x08, //                       0b001000 not shared by other get_mesh grid_dims
                break;
            case SP_ES_GHOST : // = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_outer_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                index_tuple{m_lower_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
                res.append(
                        MeshEntityIdCoder::make_range(
                                index_tuple{m_upper_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                index_tuple{m_outer_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));

                if (m_dims_[1] > 1)
                {
                    res.append(
                            MeshEntityIdCoder::make_range(
                                    index_tuple{m_lower_[0], m_outer_lower_[1], m_outer_lower_[2]},
                                    index_tuple{m_upper_[0], m_lower_[1], m_outer_upper_[2]}, entityType));
                    res.append(
                            MeshEntityIdCoder::make_range(
                                    index_tuple{m_lower_[0], m_upper_[1], m_outer_lower_[2]},
                                    index_tuple{m_upper_[0], m_outer_upper_[1], m_outer_upper_[2]}, entityType));
                }
                if (m_dims_[2] > 1)
                {
                    res.append(
                            MeshEntityIdCoder::make_range(
                                    index_tuple{m_lower_[0], m_lower_[1], m_outer_lower_[2]},
                                    index_tuple{m_upper_[0], m_upper_[1], m_lower_[2]}, entityType));
                    res.append(
                            MeshEntityIdCoder::make_range(
                                    index_tuple{m_lower_[0], m_lower_[1], m_upper_[2]},
                                    index_tuple{m_upper_[0], m_upper_[1], m_outer_upper_[2]}, entityType));
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
                    if (m_dims_[i] > 1 && m_ghost_width_[i] != 0)
                    {
                        l[i] += 1;
                        u[i] -= 1;
                    }
                }
                res.append(MeshEntityIdCoder::make_range(l, u, entityType));
                break;
            case SP_ES_OWNED:
                res.append(MeshEntityIdCoder::make_range(m_lower_, m_upper_, entityType));
                break;

            default:
                UNIMPLEMENTED;
                break;
        }
        return std::move(res);
    };


    virtual size_type max_hash(MeshEntityType entityType = VERTEX) const
    {
        return m::max_hash(m_outer_lower_, m_outer_upper_, entityType);
    }

    virtual size_type hash(MeshEntityId const &s) const
    {
        return static_cast<size_type>(m::hash(s, m_outer_lower_, m_outer_upper_));
    }


    virtual int get_adjacent_entities(MeshEntityType entity_type, MeshEntityId s,
                                      MeshEntityId *p = nullptr) const
    {
        return m::get_adjacent_entities(entity_type, entity_type, s, p);
    }


    virtual std::tuple<toolbox::DataSpace, toolbox::DataSpace>
    data_space(MeshEntityType const &t, MeshEntityStatus status = SP_ES_OWNED) const
    {
        int i_ndims = (t == EDGE || t == FACE) ? (ndims + 1) : ndims;

        nTuple<size_type, ndims + 1> f_dims, f_count;
        nTuple<size_type, ndims + 1> f_start;

        nTuple<size_type, ndims + 1> m_dims, m_count;
        nTuple<size_type, ndims + 1> m_start;

        switch (status)
        {
            case SP_ES_ALL:
                f_dims = m_shape_;//+ m_offset_;
                f_start = 0;//m_offset_;
                f_count = m_shape_;

                m_dims = m_shape_;
                m_start = 0;
                m_count = m_shape_;
                break;
            case SP_ES_OWNED:
            default:
                f_dims = m_dims_;//+ m_offset_;
                f_start = 0;//m_offset_;
                f_count = m_dims_;

                m_dims = m_shape_;
                m_start = m_lower_ - m_outer_lower_;
                m_count = m_dims_;
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

    virtual void update()
    {
        /**
             *\verbatim
             *                ^y
             *               /
             *        z     /
             *        ^    /
             *        |  110-------------111
             *        |  /|              /|
             *        | / |             / |
             *        |/  |            /  |
             *       100--|----------101  |
             *        | m |           |   |
             *        |  010----------|--011
             *        |  /            |  /
             *        | /             | /
             *        |/              |/
             *       000-------------001---> x
             *
             *\endverbatim
             */
        for (int i = 0; i < ndims; ++i)
        {

            m_dims_[i] = (m_dims_[i] <= 1) ? 1 : m_dims_[i];

            m_ghost_width_[i] = (m_dims_[i] <= 1) ? 0 : m_ghost_width_[i];

            m_shape_[i] = m_dims_[i] + m_ghost_width_[i] * 2;

            m_lower_[i] = m_ghost_width_[i];
            m_upper_[i] = m_ghost_width_[i] + m_dims_[i];

            m_outer_lower_[i] = m_lower_[i] - m_ghost_width_[i];
            m_outer_upper_[i] = m_upper_[i] + m_ghost_width_[i];

            m_inner_lower_[i] = m_lower_[i] + m_ghost_width_[i];
            m_inner_upper_[i] = m_upper_[i] - m_ghost_width_[i];
        }

        int flag = 0;

        for (int i = 0; i < 3; ++i)
        {
            if (m_dims_[0] == 1) { flag = flag | (0x3 << (i * 2)); }

            if (m_ghost_width_[i] == 0) { flag = flag | (0x1 << (i * 2)); }

        }
        status(flag);
    }


    virtual point_type point(MeshEntityId const &s) const =0;

    virtual point_type point_local_to_global(MeshEntityId s, point_type const &r) const =0;

    virtual std::tuple<MeshEntityId, point_type> point_global_to_local(point_type const &g, int nId = 0) const =0;

    virtual index_tuple point_to_index(point_type const &g, int nId = 0) const =0;

//    virtual void box(box_type const &)   =0;

    virtual box_type box(MeshEntityStatus status = SP_ES_OWNED) const =0;

    virtual index_box_type index_box(box_type const &b) const =0;

}; // struct  MeshView
}} // namespace simpla // namespace get_mesh





#endif //SIMPLA_MESHBOX_H
