//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/mesh/Worker.h>
#include <simpla/physics/Field.h>
#include <simpla/physics/FieldExpression.h>
#include <simpla/manifold/Calculus.h>

using namespace simpla;

class DummyMesh : public mesh::MeshBlock
{
public:
    static constexpr unsigned int ndims = 3;

    template<typename ...Args>
    DummyMesh(Args &&...args):mesh::MeshBlock(std::forward<Args>(args)...) {}

    ~DummyMesh() {}

    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<Real, IFORM>;

    virtual std::shared_ptr<mesh::MeshBlock> clone() const
    {
        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<DummyMesh>());
    };

    template<typename ...Args>
    Real eval(Args &&...args) const { return 1.0; };
};

template<typename TM>
struct AMRTest : public mesh::Worker
{
    typedef TM mesh_type;

    template<typename TV, mesh::MeshEntityType IFORM> using field_type=Field<TV, mesh_type, index_const<IFORM>>;
    field_type<Real, mesh::VERTEX> phi{"phi", this};
    field_type<Real, mesh::EDGE> E{"E", this};
    field_type<Real, mesh::FACE> B{"B", this};

};

int main(int argc, char **argv)
{
//    auto atlas = std::make_shared<mesh::Atlas>();
//    atlas->insert(m);

    index_type lo[3] = {0, 0, 0}, hi[3] = {40, 50, 60};
    size_type gw[3] = {0, 0, 0};

    auto m0 = std::make_shared<DummyMesh>(lo, hi, gw, 0);
    auto m1 = std::make_shared<DummyMesh>(lo, hi, gw, 1);


    auto worker = std::make_shared<AMRTest<DummyMesh>>();

    worker->E.deploy(m0.get());
    worker->E.deploy(m1.get());
    worker->B.deploy(m1.get());
    worker->E = 1.2;
    exterior_derivative(worker->phi);
//    worker->phi = diverge(worker->E);
    std::cout << *worker << std::endl;

//
//    auto m = std::make_shared<mesh::MeshBlock>();
//
//    auto attr = mesh::Attribute::create();

//    auto f = attr->create(m);
}