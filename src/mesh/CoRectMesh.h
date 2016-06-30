/**
 *
 * @file corectmesh.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <vector>
#include <iomanip>

#include "../gtl/macro.h"
#include "../sp_def.h"
#include "../gtl/nTuple.h"
#include "../gtl/nTupleExt.h"
#include "../gtl/PrettyStream.h"
#include "../gtl/type_traits.h"
#include "../gtl/type_cast.h"
#include "../gtl/Log.h"

#include "MeshCommon.h"
#include "MeshBase.h"
#include "MeshEntityId.h"

namespace simpla { namespace mesh
{
namespace tags { struct CoRectLinear; }

template<typename ...>
class Mesh;

typedef Mesh<tags::CoRectLinear> CoRectMesh;


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */
template<>
struct Mesh<tags::CoRectLinear> : public MeshBase, public MeshEntityIdCoder
{
private:
    typedef Mesh<tags::CoRectLinear> this_type;
    typedef MeshBase base_type;
public:
    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return std::string("Mesh<tags::CoRectLinear>"); }

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

    vector_type m_dx_{{1, 1, 1}}, m_inv_dx_{{1, 1, 1}}; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

    index_tuple m_dims_{{10, 10, 10}}, m_shape_{{10, 10, 10}};

    index_tuple m_lower_{{0, 0, 0}}, m_upper_{{10, 10, 10}};

    index_tuple m_inner_lower_{{0, 0, 0}}, m_inner_upper_{{10, 10, 10}};

    index_tuple m_outer_lower_{{0, 0, 0}}, m_outer_upper_{{10, 10, 10}};

    index_tuple m_ghost_width_{{0, 0, 0}};

    index_tuple m_offset_{{0, 0, 0}};

    typedef MeshEntityIdCoder m;

    typedef MeshEntityId id_type;

public:
    static constexpr int ndims = 3;

    Mesh() { }

    Mesh(this_type const &other)
    {
        m_dims_ = other.m_dims_;
        m_coords_lower_ = other.m_coords_lower_;
        m_coords_upper_ = other.m_coords_upper_;
        deploy();
    };

    virtual  ~Mesh() { }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {

        os
        << std::setw(indent) << " "
        << "Topology = { Type = \"CoRectMesh\", "
        << "Dimensions = " << dimensions() << " offset = " << offset() << " }," << std::endl
        << std::setw(indent) << " " << "Box = " << box() << "," << std::endl;

        return os;
    }

    template<typename TDict>
    void setup(TDict const &dict)
    {
        dimensions(dict["Dimensions"].template as<index_tuple>(index_tuple{20, 20, 1}));
        box(dict["Box"].template as<box_type>(box_type{{0, 0, 0},
                                                       {1, 1, 1}}));
        deploy();
    }

    virtual io::IOStream &save(io::IOStream &os) const
    {
        os.open(type_cast<std::string>(this->short_id()) + "/");
        os.set_attribute(".dims", dimensions());
        os.set_attribute(".box", box());
        return os;
    };


    void dimensions(index_tuple const &d) { m_dims_ = d; }

    index_tuple const &dimensions() const { return m_dims_; }

    void offset(index_tuple const &d) { m_offset_ = d; }

    virtual index_tuple offset() const { return m_offset_; }

    virtual point_type origin_point() const { return m_coords_lower_; };

    void ghost_width(index_tuple const &d) { m_ghost_width_ = d; }

    index_tuple const &ghost_width() const { return m_ghost_width_; }

    template<typename X0, typename X1>
    void box(X0 const &x0, X1 const &x1)
    {
        m_coords_lower_ = x0;
        m_coords_upper_ = x1;
    }


    void box(box_type const &b) { std::tie(m_coords_lower_, m_coords_upper_) = b; }

    vector_type const &dx() const { return m_dx_; }

    virtual std::shared_ptr<MeshBase> clone() const
    {
        return std::make_shared<this_type>(*this);
    }

    virtual std::shared_ptr<MeshBase> extend(int const *od, size_type w = 2) const
    {

        index_tuple dims{0, 0, 0};
        point_type lower{0, 0, 0};
        point_type upper{0, 0, 0};

        bool valid = true;

        int flag = status();

        for (int i = 0; i < 3; ++i)
        {
            if (m_dims_[i] == 1 && od[i] != 0)
            {
                valid = false;
            }
            else
            {
                switch (od[i])
                {
                    case 1:
                        dims[i] = w;
                        lower[i] = m_coords_upper_[i];
                        upper[i] = m_coords_upper_[i] + m_dx_[i] * w;
                        break;
                    case -1:
                        dims[i] = w;
                        lower[i] = m_coords_lower_[i] - m_dx_[i] * w;
                        upper[i] = m_coords_lower_[i];
                        break;
                    case 0:
                        dims[i] = m_dims_[i];
                        lower[i] = m_coords_lower_[i];
                        upper[i] = m_coords_upper_[i];
                        break;
                    default:
                        valid = false;
                        break;
                }
            }
        }
        if (valid)
        {
            auto res = std::dynamic_pointer_cast<this_type>(this->clone());
            res->dimensions(dims);
            res->box(std::make_tuple(lower, upper));
            res->ghost_width(ghost_width());
            res->deploy();
            return std::dynamic_pointer_cast<MeshBase>(res);
        } else
        {
            return nullptr;
        }
    };

private:
    //TODO should use block-entity_id_range
    parallel::concurrent_unordered_set <MeshEntityId> m_affected_entities_[4];
    parallel::concurrent_unordered_set <MeshEntityId> m_interface_entities_[4];
public:

    typedef typename MeshEntityIdCoder::range_type block_range_type;

    virtual MeshEntityRange select(box_type const &other,
                                   MeshEntityType entityType = VERTEX,
                                   MeshEntityStatus status = SP_ES_VALID) const
    {

        point_type c_lower, c_upper;
        std::tie(c_lower, c_upper) = box(status);

        bool overlapped = true;

        for (int i = 0; i < 3; ++i)
        {
            c_lower[i] = std::max(c_lower[i], std::get<0>(other)[i]);
            c_upper[i] = std::min(c_upper[i], std::get<1>(other)[i]);

            if (c_lower[i] >= c_upper[i]) { overlapped = false; }
        }

        if (!overlapped)
        {
            return MeshEntityRange();
        }
        else
        {
            return MeshEntityRange(
                    MeshEntityIdCoder::make_range(point_to_index(c_lower), point_to_index(c_upper), entityType));
        }

    };

    virtual box_type box(MeshEntityStatus status = SP_ES_VALID) const
    {
        box_type res;

        switch (status)
        {
            case SP_ES_VALID : //all valid
                std::get<0>(res) = m_coords_lower_ - m_dx_ * m_ghost_width_;
                std::get<1>(res) = m_coords_upper_ + m_dx_ * m_ghost_width_;;
                break;
            case SP_ES_LOCAL : //local and valid
                std::get<0>(res) = m_coords_lower_ + m_dx_ * m_ghost_width_;;
                std::get<1>(res) = m_coords_upper_ - m_dx_ * m_ghost_width_;
                break;
            case SP_ES_OWNED:
                std::get<0>(res) = m_coords_lower_;
                std::get<1>(res) = m_coords_upper_;
                break;
            case SP_ES_INTERFACE: //SP_ES_INTERFACE
            case SP_ES_GHOST : //local and valid
            default:
                UNIMPLEMENTED;
                break;


        }
        return std::move(res);
    }

    virtual MeshEntityRange range(box_type const &b, MeshEntityType entityType = VERTEX) const
    {
        point_type x_lower, x_upper;

        index_tuple lower, upper;
        std::tie(x_lower, x_upper) = b;
        for (int i = 0; i < 3; ++i)
        {
            if (m_dims_[i] > 1)
            {
                lower[i] = m_lower_[i] + static_cast<index_type >((x_lower[i] - m_coords_lower_[i]) / m_dx_[i]);
                upper[i] = m_lower_[i] + static_cast<index_type >((x_upper[i] - m_coords_lower_[i]) / m_dx_[i]);
            }
            else
            {
                lower[i] = m_lower_[i];
                upper[i] = m_upper_[i];
            }
        }

        return MeshEntityIdCoder::make_range(lower, upper, entityType);
    }

    virtual MeshEntityRange range(MeshEntityType entityType = VERTEX, MeshEntityStatus status = SP_ES_VALID) const
    {
        MeshEntityRange res;

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
            case SP_ES_VALID : //all valid
                MeshEntityRange(MeshEntityIdCoder::make_range(m_lower_, m_upper_, entityType)).swap(res);
                break;
            case SP_ES_NON_LOCAL : // = SP_ES_SHARED | SP_ES_OWNED, //              0b000101
            case SP_ES_SHARED : //       = 0x04,                    0b000100 shared by two or more get_mesh blocks
                break;
            case SP_ES_NOT_SHARED  : // = 0x08, //                       0b001000 not shared by other get_mesh blocks
                break;

            case SP_ES_GHOST : // = SP_ES_SHARED | SP_ES_NOT_OWNED, //              0b000110

            case SP_ES_DMZ: //  = 0x100,
            case SP_ES_NOT_DMZ: //  = 0x200,

            case SP_ES_LOCAL : // = SP_ES_NOT_SHARED | SP_ES_OWNED, //              0b001001
                MeshEntityRange(MeshEntityIdCoder::make_range(m_inner_lower_, m_inner_upper_, entityType)).swap(res);
                break;


            case SP_ES_INTERFACE: //  = 0x010, //                        0b010000 interface(boundary) shared by two get_mesh blocks,
                MeshEntityRange(m_interface_entities_[entityType]).swap(res);
                break;


            default:
                UNIMPLEMENTED;
                break;
        }
        return std::move(res);
    };


    virtual size_t max_hash(MeshEntityType entityType = VERTEX) const
    {
        return m::max_hash(m_outer_lower_, m_outer_upper_, entityType);
    }

    virtual size_t hash(MeshEntityId const &s) const
    {
        return static_cast<size_t>(m::hash(s, m_outer_lower_, m_outer_upper_));
    }


    virtual point_type
    point(MeshEntityId const &s) const
    {
        point_type p = m::point(s);

        p[0] = std::fma(p[0], m_l2g_scale_[0], m_l2g_shift_[0]);
        p[1] = std::fma(p[1], m_l2g_scale_[1], m_l2g_shift_[1]);
        p[2] = std::fma(p[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(p);

    }

    virtual point_type
    point_local_to_global(MeshEntityId s, point_type const &r) const
    {
        point_type p = m::point_local_to_global(s, r);

        p[0] = std::fma(p[0], m_l2g_scale_[0], m_l2g_shift_[0]);
        p[1] = std::fma(p[1], m_l2g_scale_[1], m_l2g_shift_[1]);
        p[2] = std::fma(p[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(p);
    }

    virtual std::tuple<MeshEntityId, point_type>
    point_global_to_local(point_type const &g, int nId = 0) const
    {

        return m::point_global_to_local(
                point_type{
                        std::fma(g[0], m_g2l_scale_[0], m_g2l_shift_[0]),
                        std::fma(g[1], m_g2l_scale_[1], m_g2l_shift_[1]),
                        std::fma(g[2], m_g2l_scale_[2], m_g2l_shift_[2])
                }, nId);
    }

    virtual index_tuple
    point_to_index(point_type const &g, int nId = 0) const
    {
        return m::unpack_index(std::get<0>(m::point_global_to_local(
                point_type{
                        std::fma(g[0], m_g2l_scale_[0], m_g2l_shift_[0]),
                        std::fma(g[1], m_g2l_scale_[1], m_g2l_shift_[1]),
                        std::fma(g[2], m_g2l_scale_[2], m_g2l_shift_[2])
                }, nId)));
    };

    vector_type m_l2g_scale_{{1, 1, 1}}, m_l2g_shift_{{0, 0, 0}};
    vector_type m_g2l_scale_{{1, 1, 1}}, m_g2l_shift_{{0, 0, 0}};

    virtual int get_adjacent_entities(MeshEntityType const &entity_type, MeshEntityId const &s,
                                      MeshEntityId *p = nullptr) const
    {
        return m::get_adjacent_entities(entity_type, entity_type, s, p);
    }

    virtual std::shared_ptr<MeshBase> refine(box_type const &b, int flag = 0) const
    {
        return std::shared_ptr<MeshBase>();
    }


    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(id_type s) const { return m_volume_[m::node_id(s)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_[m::node_id(s)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_[m::node_id(s)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_[m::node_id(s)]; }

    void deploy()
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

            m_dims_[i] = (m_dims_[i] > 0) ? m_dims_[i] : 1;

            m_dx_[i] = (m_coords_upper_[i] - m_coords_lower_[i]) / static_cast<Real>( m_dims_[i]);

            m_inv_dx_[i] = static_cast<Real>(1.0) / m_dx_[i];

            m_ghost_width_[i] = (m_dims_[i] > 1) ? m_ghost_width_[i] : 0;

            m_lower_[i] = m_ghost_width_[i];

            m_upper_[i] = m_lower_[i] + m_dims_[i];

            m_shape_[i] = m_upper_[i] + m_ghost_width_[i];

            m_inner_lower_[i] = m_lower_[i] + m_ghost_width_[i];
            m_inner_upper_[i] = m_upper_[i] - m_ghost_width_[i];

            m_outer_lower_[i] = 0;
            m_outer_upper_[i] = m_shape_[i];

            m_l2g_scale_[i] = (m_dims_[i] <= 1) ? 0 : m_dx_[i];
            m_l2g_shift_ = m_coords_lower_;

            m_g2l_scale_[i] = (m_dims_[i] <= 1) ? 0 : static_cast<Real>(1.0) / m_dx_[i];
            m_g2l_shift_[i] = (m_dims_[i] <= 1) ? 0 : -m_coords_lower_[i] * m_g2l_scale_[i];

        }


        m_volume_[0 /*000*/] = 1;
        m_volume_[1 /*001*/] = m_dx_[0];
        m_volume_[2 /*010*/] = m_dx_[1];
        m_volume_[4 /*100*/] = m_dx_[2];
        m_volume_[3 /*011*/] = m_dx_[0] * m_dx_[1];
        m_volume_[5 /*101*/] = m_dx_[2] * m_dx_[0];
        m_volume_[6 /*110*/] = m_dx_[1] * m_dx_[2];
        m_volume_[7 /*110*/] = m_dx_[0] * m_dx_[1] * m_dx_[2];


        m_dual_volume_[0 /*000*/] = m_volume_[7];
        m_dual_volume_[1 /*001*/] = m_volume_[6];
        m_dual_volume_[2 /*010*/] = m_volume_[5];
        m_dual_volume_[4 /*100*/] = m_volume_[3];
        m_dual_volume_[3 /*011*/] = m_volume_[4];
        m_dual_volume_[5 /*101*/] = m_volume_[2];
        m_dual_volume_[6 /*110*/] = m_volume_[1];
        m_dual_volume_[7 /*110*/] = m_volume_[0];


        m_inv_dx_[0] = (m_dims_[0] == 1) ? 1 : 1 / m_dx_[0];
        m_inv_dx_[1] = (m_dims_[1] == 1) ? 1 : 1 / m_dx_[1];
        m_inv_dx_[2] = (m_dims_[2] == 1) ? 1 : 1 / m_dx_[2];


        m_inv_volume_[0 /*000*/] = 1;
        m_inv_volume_[1 /*001*/] = m_inv_dx_[0];
        m_inv_volume_[2 /*010*/] = m_inv_dx_[1];
        m_inv_volume_[4 /*100*/] = m_inv_dx_[2];
        m_inv_volume_[3 /*011*/] = m_inv_dx_[0] * m_inv_dx_[1];
        m_inv_volume_[5 /*101*/] = m_inv_dx_[2] * m_inv_dx_[0];
        m_inv_volume_[6 /*110*/] = m_inv_dx_[1] * m_inv_dx_[2];
        m_inv_volume_[7 /*110*/] = m_inv_dx_[0] * m_inv_dx_[1] * m_inv_dx_[2];


        m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
        m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
        m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
        m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
        m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
        m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
        m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
        m_inv_dual_volume_[7 /*110*/] = m_inv_volume_[0];


        m_inv_dx_[0] = (m_dims_[0] == 1) ? 0 : m_inv_dx_[0];
        m_inv_dx_[1] = (m_dims_[1] == 1) ? 0 : m_inv_dx_[1];
        m_inv_dx_[2] = (m_dims_[2] == 1) ? 0 : m_inv_dx_[2];


        m_inv_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 0 : m_inv_dx_[0];
        m_inv_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 0 : m_inv_dx_[1];
        m_inv_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 0 : m_inv_dx_[2];


        int flag = 0;

        for (int i = 0; i < 3; ++i)
        {
            if (m_dims_[0] == 1)
            {
                flag = flag | (0x3 << (i * 2));
            }

            if (m_ghost_width_[i] == 0)
            {
                flag = flag | (0x1 << (i * 2));
            }

        }
        status(flag);
    }


    virtual std::tuple<data_model::DataSpace, data_model::DataSpace> data_space(MeshEntityType const &t) const
    {
        int i_ndims = (t == EDGE || t == FACE) ? (ndims + 1) : ndims;

        nTuple<size_t, ndims + 1> f_dims, f_count;
        nTuple<ptrdiff_t, ndims + 1> f_start;

        nTuple<size_t, ndims + 1> m_dims, m_count;
        nTuple<ptrdiff_t, ndims + 1> m_start;

        f_dims = m_dims_ + m_offset_;
        f_start = m_offset_;
        f_count = m_dims_;
        f_dims[ndims] = 3;
        f_start[ndims] = 0;
        f_count[ndims] = 3;


        data_model::DataSpace f_space(i_ndims, &f_dims[0]);
        f_space.select_hyperslab(&f_start[0], nullptr, &f_count[0], nullptr);

        m_dims = m_shape_;
        m_start = m_lower_;
        m_count = m_dims_;
        m_dims[ndims] = 3;
        m_start[ndims] = 0;
        m_count[ndims] = 3;


        data_model::DataSpace m_space(i_ndims, &m_dims[0]);
        m_space.select_hyperslab(&m_start[0], nullptr, &m_count[0], nullptr);

        return std::forward_as_tuple(m_space, f_space);

    };


//    int get_vertices(int node_id, id_type s, point_type *p = nullptr) const
//    {
//
//        int num = m::get_adjacent_entities(VERTEX, node_id, s);
//
//        if (p != nullptr)
//        {
//            id_type neighbour[num];
//
//            m::get_adjacent_entities(VERTEX, node_id, s, neighbour);
//
//            for (int i = 0; i < num; ++i)
//            {
//                p[i] = point(neighbour[i]);
//            }
//
//        }
//
//
//        return num;
//    }

/**
 * @name  Coordinate map
 * @{
 *
 *        Topology mesh       geometry get_mesh
 *                        map
 *              M      ---------->      G
 *              x                       y
 **/
//private:
//
//
//    point_type m_l2g_shift_ = {0, 0, 0};
//
//    point_type m_l2g_scale_ = {1, 1, 1};
//
//    point_type m_g2l_shift_ = {0, 0, 0};
//
//    point_type m_g2l_scale_ = {1, 1, 1};
//
//
//    point_type inv_map(point_type const &x) const
//    {
//
//        point_type res;
//
//        res[0] = std::fma(x[0], m_g2l_scale_[0], m_g2l_shift_[0]);
//
//        res[1] = std::fma(x[1], m_g2l_scale_[1], m_g2l_shift_[1]);
//
//        res[2] = std::fma(x[2], m_g2l_scale_[2], m_g2l_shift_[2]);
//
//        return std::move(res);
//    }
//
//    point_type map(point_type const &y) const
//    {
//
//        point_type res;
//
//
//        res[0] = std::fma(y[0], m_l2g_scale_[0], m_l2g_shift_[0]);
//
//        res[1] = std::fma(y[1], m_l2g_scale_[1], m_l2g_shift_[1]);
//
//        res[2] = std::fma(y[2], m_l2g_scale_[2], m_l2g_shift_[2]);
//
//        return std::move(res);
//    }
//
//public:
//
//    virtual point_type point(id_type const &s) const { return std::move(map(m::point(s))); }
//
//    virtual point_type point_local_to_global(id_type s, point_type const &x) const
//    {
//        return std::move(map(m::point_local_to_global(s, x)));
//    }
//
//    virtual point_type point_local_to_global(std::tuple<id_type, point_type> const &t) const
//    {
//        return std::move(map(m::point_local_to_global(t)));
//    }
//
//    virtual std::tuple<id_type, point_type> point_global_to_local(point_type const &x, int n_id = 0) const
//    {
//        return std::move(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//    virtual id_type id(point_type const &x, int n_id = 0) const
//    {
//        return std::get<0>(m::point_global_to_local(inv_map(x), n_id));
//    }
//
//
//
//    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b) const
//    {
//
//        point_type b0, b1, x0, x1;
//
//        std::tie(b0, b1) = local_index_box();
//
//        std::tie(x0, x1) = b;
//
//        if (geometry::box_intersection(b0, b1, &x0, &x1))
//        {
//            return std::make_tuple(m::unpack_index(id(x0)),
//                                   m::unpack_index(id(x1) + (m::_DA << 1)));
//
//        }
//        else
//        {
//            index_tuple i0, i1;
//            i0 = 0;
//            i1 = 0;
//            return std::make_tuple(i0, i1);
//        }
//
//    }


//    struct calculus_policy
//    {
//        template<typename ...Args> static double eval(Args &&...args) { return 1.0; }
//    };
}; // struct  Mesh
}} // namespace simpla // namespace get_mesh

#endif //SIMPLA_CORECTMESH_H
