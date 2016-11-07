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

    SP_OBJECT_HEAD(DummyMesh, mesh::MeshBlock)

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

    SP_OBJECT_HEAD(AMRTest, mesh::Worker);

    template<typename TV, mesh::MeshEntityType IFORM> using field_type=Field<TV, mesh_type, index_const<IFORM>>;
    field_type<Real, mesh::VERTEX> phi{"phi", this};
    field_type<Real, mesh::EDGE> E{"E", this};
    field_type<Real, mesh::FACE> B{"B", this};

};

int main(int argc, char **argv)
{

    index_type lo[3] = {0, 0, 0}, hi[3] = {40, 50, 60};
    index_type lo1[3] = {10, 20, 30}, hi1[3] = {20, 30, 40};

    size_type gw[3] = {0, 0, 0};

    auto atlas = std::make_shared<mesh::Atlas>();
    auto m0 = atlas->add<DummyMesh>("FirstLevel", lo, hi, gw, 0);
    auto m1 = atlas->refine(m0, lo1, hi1);

    std::cout << *atlas << std::endl;

    auto worker = std::make_shared<AMRTest<DummyMesh>>();
    worker->move_to(m0);
    TRY_CALL(worker->deploy());
    worker->move_to(m1);
    TRY_CALL(worker->deploy());
    worker->E = 1.2;
    worker->E = exterior_derivative(worker->phi);
//    worker->phi = diverge(worker->E);
    std::cout << " Worker = {" << *worker << "}" << std::endl;
    std::cout << "E = {" << worker->E << "}" << std::endl;
    std::cout << "E = {" << *worker->E.attribute() << "}" << std::endl;
//
//    auto m = std::make_shared<mesh::MeshBlock>();
//
//    auto attr = mesh::Attribute::clone();

//    auto f = attr->clone(m);
}