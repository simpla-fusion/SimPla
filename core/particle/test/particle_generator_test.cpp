/**
 * @file particle_generator.cpp
 * @author salmon
 * @date 2015-11-28.
 */
#include <tbb/task_group.h>
#include <tbb/concurrent_vector.h>
#include "../particle_engine.h"
#include "../../dataset/datatype_ext.h"
#include "../../io/io.h"
#include "Parallel.h"
#include "../particle_generator.h"

namespace simpla
{
struct PICDemo
{
    SP_DEFINE_STRUCT(point_type,
                     Vec3, x,
                     Vec3, v
    );

    SP_DEFINE_STRUCT(sample_type,
                     point_type, z,
                     Real, f,
                     Real, w
    );

    SP_DEFINE_PROPERTIES(
            Real, mass,
            Real, charge,
            Real, temperature
    )

    void update() { }

    Vec3 project(sample_type const &p) const { return p.z.x; }

    Real function_value(sample_type const &p) const { return p.f * p.w; }


    point_type lift(Vec3 const &x, Vec3 const &v) const
    {
        return point_type{x, v};
    }

    point_type lift(std::tuple<Vec3, Vec3> const &z) const
    {
        return point_type{std::get<0>(z), std::get<1>(z)};
    }

    sample_type sample(Vec3 const &x, Vec3 const &v, Real f) const
    {
        return sample_type{lift(x, v), f, 0};
    }

    sample_type sample(point_type const &z, Real f) const
    {
        return sample_type{z, f, 0};
    }

    template<typename TE, typename TB>
    void push(sample_type *p, Real dt, Real t, TE const &E, TB const &B)
    {
        p->z.x += p->z.v * dt * 0.5;
        p->z.v += E(p->z.x) * dt;
        p->z.x += p->z.v * dt * 0.5;
    };

};
}//namespace simpla

using namespace simpla;


int main(int argc, char **argv)
{
    logger::init(argc, argv);

    parallel::init(argc, argv);

    io::init(argc, argv);

    PICDemo pic_engine;

    auto box = std::make_tuple(nTuple<Real, 3>{0, 0, 0},
                               nTuple<Real, 3>{1, 1, 1});

    ParticleGenerator<PICDemo> gen(pic_engine);

    gen.reserve(1000);

    tbb::concurrent_vector<PICDemo::sample_type> data;

    auto fun = [&]()
    {
        int pic = 100;

        auto it_range = gen.generator(pic, box, 1.0);

        std::copy(std::get<0>(it_range), std::get<1>(it_range), data.grow_by(pic));
    };

    tbb::task_group group;
    for (int i = 0; i < 10; ++i)
    {
        group.run([&]() { fun(); });
    }


    group.wait();

    std::vector<PICDemo::sample_type> data2;

    std::copy(data.begin(), data.end(), std::back_inserter(data2));

    LOGGER << io::write("D", traits::make_dataset(&data2[0], data2.size())) << std::endl;
}