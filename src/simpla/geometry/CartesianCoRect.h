//
// Created by salmon on 16-11-17.
//

#ifndef SIMPLA_CARTESIANCORECT_H
#define SIMPLA_CARTESIANCORECT_H


#include <vector>
#include <iomanip>

#include <simpla/mesh/DataBlock.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/EntityId.h>

namespace simpla { namespace geometry
{

struct CartesianCoRect : public mesh::MeshBlock
{

public:
    static constexpr unsigned int NDIMS = 3;

    SP_OBJECT_HEAD(CartesianCoRect, mesh::MeshBlock)

    typedef Real scalar_type;

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



public:

    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<TV, IFORM>;

    CartesianCoRect() {}

    template<typename ...Args>
    CartesianCoRect(Args &&...args): MeshBlock(std::forward<Args>(args)...) {};

    CartesianCoRect(CartesianCoRect const &other) : MeshBlock(other)
    {

        for (int i = 0; i < 9; ++i)
        {
            m_volume_[i] = m_volume_[i];
            m_inv_volume_[i] = m_inv_volume_[i];
            m_dual_volume_[i] = m_dual_volume_[i];
            m_inv_dual_volume_[i] = m_inv_dual_volume_[i];
        }


    };

    virtual  ~CoRectMesh() {}

    virtual void deploy();

    virtual std::shared_ptr<mesh::MeshBlock> clone() const { return std::make_shared<CartesianCoRect>(*this); };


private:
    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(mesh::MeshEntityId s) const { return m_volume_[m::node_id(s)]; }

    virtual Real dual_volume(mesh::MeshEntityId s) const { return m_dual_volume_[m::node_id(s)]; }

    virtual Real inv_volume(mesh::MeshEntityId s) const { return m_inv_volume_[m::node_id(s)]; }

    virtual Real inv_dual_volume(mesh::MeshEntityId s) const { return m_inv_dual_volume_[m::node_id(s)]; }


}; // struct  Mesh

void CartesianCoRect::deploy()
{
    mesh::MeshBlock::deploy();
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
    auto const &dims = dimensions();


    m_volume_[0 /*000*/] = 1;
    m_volume_[1 /*001*/] = (dims[0] == 1) ? 1 : m_dx_[0];
    m_volume_[2 /*010*/] = (dims[1] == 1) ? 1 : m_dx_[1];
    m_volume_[4 /*100*/] = (dims[2] == 1) ? 1 : m_dx_[2];
    m_volume_[3 /*011*/] = m_volume_[1] * m_volume_[2];
    m_volume_[5 /*101*/] = m_volume_[4] * m_volume_[1];
    m_volume_[6 /*110*/] = m_volume_[4] * m_volume_[2];
    m_volume_[7 /*111*/] = m_volume_[1] * m_volume_[2] * m_volume_[4];


    m_dual_volume_[0 /*000*/] = m_volume_[7];
    m_dual_volume_[1 /*001*/] = m_volume_[6];
    m_dual_volume_[2 /*010*/] = m_volume_[5];
    m_dual_volume_[4 /*100*/] = m_volume_[3];
    m_dual_volume_[3 /*011*/] = m_volume_[4];
    m_dual_volume_[5 /*101*/] = m_volume_[2];
    m_dual_volume_[6 /*110*/] = m_volume_[1];
    m_dual_volume_[7 /*111*/] = m_volume_[0];


    m_inv_volume_[0 /*000*/] = 1;
    m_inv_volume_[1 /*001*/] = (dims[0] == 1) ? 1 : m_inv_dx_[0];
    m_inv_volume_[2 /*010*/] = (dims[1] == 1) ? 1 : m_inv_dx_[1];
    m_inv_volume_[4 /*100*/] = (dims[2] == 1) ? 1 : m_inv_dx_[2];
    m_inv_volume_[3 /*011*/] = m_inv_volume_[2] * m_inv_volume_[1];
    m_inv_volume_[5 /*101*/] = m_inv_volume_[4] * m_inv_volume_[1];
    m_inv_volume_[6 /*110*/] = m_inv_volume_[4] * m_inv_volume_[2];
    m_inv_volume_[7 /*111*/] = m_inv_volume_[1] * m_inv_volume_[2] * m_inv_volume_[4];


    m_inv_volume_[1 /*001*/] = (dims[0] == 1) ? 0 : m_inv_volume_[1];
    m_inv_volume_[2 /*010*/] = (dims[1] == 1) ? 0 : m_inv_volume_[2];
    m_inv_volume_[4 /*100*/] = (dims[2] == 1) ? 0 : m_inv_volume_[4];


    m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
    m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
    m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
    m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
    m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
    m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
    m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
    m_inv_dual_volume_[7 /*111*/] = m_inv_volume_[0];


}
}}
#endif //SIMPLA_CARTESIANCORECT_H
