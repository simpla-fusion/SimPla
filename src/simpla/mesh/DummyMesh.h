/** 
 * @file DummyMesh.h
 * @author salmon
 * @date 16-5-25 - 上午8:25
 *  */

#ifndef SIMPLA_DUMMYMESH_H
#define SIMPLA_DUMMYMESH_H

#include <memory>

#include <simpla/SIMPLA_config.h>
#include "MeshCommon.h"
#include "MeshBlock.h"
#include "DataBlock.h"
#include "EntityIdRange.h"

namespace simpla { namespace mesh
{
class DummyMesh : public mesh::MeshBlock
{
public:
    static constexpr unsigned int ndims = 3;

    SP_OBJECT_HEAD(DummyMesh, mesh::MeshBlock)

    template<typename ...Args>
    DummyMesh(Args &&...args):mesh::MeshBlock(std::forward<Args>(args)...) {}

    ~DummyMesh() {}

    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<Real, IFORM>;

    virtual std::shared_ptr<mesh::MeshBlock> clone() const
    {
        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<DummyMesh>());
    };

    template<typename TV, mesh::MeshEntityType IFORM>
    std::shared_ptr<mesh::DataBlock> create_data_block(void *p) const
    {
        auto b = outer_index_box();

        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::get<0>(b)[2], 0};
        index_type hi[4] = {std::get<1>(b)[0], std::get<1>(b)[1], std::get<0>(b)[2], 3};
        return std::dynamic_pointer_cast<mesh::DataBlock>(
                std::make_shared<data_block_type<TV, IFORM>>(
                        static_cast<TV *>(p),
                        (IFORM == mesh::VERTEX || IFORM == mesh::VOLUME) ? 3 : 4,
                        lo, hi));
    };


    template<typename ...Args>
    Real eval(Args &&...args) const { return 1.0; };
};

}}//namespace simpla { namespace get_mesh

#endif //SIMPLA_DUMMYMESH_H
