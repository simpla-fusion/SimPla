//
// Created by salmon on 17-8-9.
//

#include "simpla/particle/Particle.h"
#include "simpla/engine/Mesh.h"
#include "simpla/geometry/csCartesian.h"
#include "simpla/mesh/RectMesh.h"
#include "simpla/particle/ParticleInitialLoad.h"
#include "simpla/scheme/FVM.h"
using namespace simpla;
using namespace simpla::data;
typedef engine::Mesh<geometry::csCartesian, mesh::RectMesh, scheme::FVM> DummyMesh;
int main(int argc, char** argv) {
    DummyMesh m;

    Particle<DummyMesh> p(&m, 4, "m"_ = 1.0, "q"_ = -1.0);

    int rnd_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM,
                       SP_RAND_NORMAL,  SP_RAND_NORMAL,  SP_RAND_NORMAL};

    //    ParticleInitialLoad(p.Get(), p.GetSize(), 6, rnd_type, 0);
    p.db()->Serialize(std::cout, 0);
    //  p.Load(200);
}