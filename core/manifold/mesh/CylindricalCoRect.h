/**
 * @file CylindricalRectMesh.h
 * @author salmon
 * @date 2016-01-11.
 */

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H

#include <limits>

#include "../mesh/MeshBlock.h"
#include "../../geometry/csCylindrical.h"
#include "../../gtl/utilities/memory_pool.h"

namespace simpla { namespace mesh
{


class CylindricalCoRect : public geometry::CylindricalMetric, public MeshBlock
{

private:
    typedef CylindricalCoRect this_type;
public:

    SP_OBJECT_HEAD(CylindricalCoRect, MeshBlock)

    typedef geometry::CylindricalMetric metric_type;

    typedef typename metric_type::cs coordinate_system_type;

    typedef MeshBlock block_type;

    using block_type::ndims;
    using block_type::id_type;
    using block_type::id_tuple;
    using block_type::index_type;
    using block_type::index_tuple;
    using block_type::range_type;
    using block_type::difference_type;
    using block_type::index_box;

    using typename metric_type::scalar_type;
    using typename metric_type::point_type;
    using typename metric_type::vector_type;
    typedef std::tuple<point_type, point_type> box_type;


public:


    CylindricalCoRect() : block_type() { }

    virtual  ~CylindricalCoRect() { }

    CylindricalCoRect(this_type const &other) = delete;

    this_type &operator=(this_type const &other) = delete;

    virtual void deploy();


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {

        os
        << std::setw(indent) << "\tGeometry={" << std::endl
        << std::setw(indent) << "\t\t Topology = { Type = \"RectMesh\",  }," << std::endl
        << std::setw(indent) << "\t\t Box = {" << box() << "}," << std::endl
        << std::setw(indent) << "\t\t Dimensions = " << block_type::dimensions() << "," << std::endl
        << std::setw(indent) << "\t\t}, " << std::endl
        << std::setw(indent) << "\t}" << std::endl;

        return os;
    }


private:
    std::vector<Real> m_volume_;
    std::vector<Real> m_dual_volume_;
    std::vector<Real> m_inv_volume_;
    std::vector<Real> m_inv_dual_volume_;

    index_type m_min_index_r_ = 0;
    index_type m_max_index_r_ = 1;

    size_t hash_(id_type s) const
    {
        return (block_type::unpack_index(s, 0) - m_min_index_r_) * block_type::NUM_OF_NODE_ID +
               block_type::node_id(s);
    }

public:
    virtual Real volume(id_type s) const
    {
        return m_volume_[hash_(s)];
    }

    virtual Real dual_volume(id_type s) const
    {
        return m_dual_volume_[hash_(s)];
    }

    virtual Real inv_volume(id_type s) const
    {

        return m_inv_volume_[hash_(s)];
    }

    virtual Real inv_dual_volume(id_type s) const
    {
        return m_inv_dual_volume_[hash_(s)];
    }




//    template<typename TGeo> void update_volume(TGeo const &geo);


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

};//struct RectMesh

void CylindricalCoRect::deploy()
{


    block_type::deploy();


    auto dims = dimensions();

    auto m_dx_ = dx();

    // update volume
    auto mem_index_box = block_type::memory_index_box();

    m_min_index_r_ = std::get<0>(mem_index_box)[0];
    m_max_index_r_ = std::get<1>(mem_index_box)[0];

    index_type num = (m_max_index_r_ - m_min_index_r_) * block_type::NUM_OF_NODE_ID;

    m_volume_.resize(num);
    m_inv_volume_.resize(num);
    m_dual_volume_.resize(num);
    m_inv_dual_volume_.resize(num);

    for (index_type i = m_min_index_r_; i < m_max_index_r_; ++i)
    {


        size_t n = (i - m_min_index_r_) * block_type::NUM_OF_NODE_ID;

        block_type::get_element_volume_in_cell(*this, block_type::pack(i, 0, 0),
                                               &m_volume_[n],
                                               &m_inv_volume_[n],
                                               &m_dual_volume_[n],
                                               &m_inv_dual_volume_[n]);
        Real r0 = point(block_type::pack(i, 0, 0))[0];
        Real r1 = point(block_type::pack(i + 1, 0, 0))[0];

        if (dims[0] <= 1) { r1 = r0; }
        /**
        *\verbatim
        *                ^Z
        *               /
        *       Theta  /
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
        *       000-------------001---> R
        *
        *\endverbatim
        */

        m_volume_[n + 4] *= r0;
        m_volume_[n + 5] *= (r0 + r1) * 0.5;
        m_volume_[n + 6] *= r0;
        m_volume_[n + 7] *= (r0 + r1) * 0.5;

        m_inv_volume_[n + 4] /= r0;
        m_inv_volume_[n + 5] /= (r0 + r1) * 0.5;
        m_inv_volume_[n + 6] /= r0;
        m_inv_volume_[n + 7] /= (r0 + r1) * 0.5;

        m_dual_volume_[n + 0] *= r0;
        m_dual_volume_[n + 1] *= (r0 + r1) * 0.5;
        m_dual_volume_[n + 2] *= r0;
        m_dual_volume_[n + 3] *= (r0 + r1) * 0.5;

        m_inv_dual_volume_[n + 0] /= r0;
        m_inv_dual_volume_[n + 1] /= (r0 + r1) * 0.5;
        m_inv_dual_volume_[n + 2] /= r0;
        m_inv_dual_volume_[n + 3] /= (r0 + r1) * 0.5;
    }


}

}}  // namespace mesh // namespace simpla
#endif //SIMPLA_CYLINDRICALRECTMESH_H
