/**
 * @file collision_pic.cpp
 * @author salmon
 * @date 2015-11-11.
 */

#include "collision_pic.h"

#include <stddef.h>
#include <iostream>
#include <memory>
#include <string>

#include "../../core/application/application.h"
#include "../../core/gtl/primitives.h"
#include "../../core/gtl/type_cast.h"
#include "../../core/io/io.h"
#include "../../core/io/io_ext.h"

#include "../../core/physics/constants.h"
#include "../../core/physics/physical_constants.h"
#include "../../core/gtl/utilities/config_parser.h"
#include "../../core/gtl/utilities/log.h"
#include "../../core/particle/obsolete/simple_particle_generator.h"


using namespace simpla;

SP_DEFINE_STRUCT(pic_mark,
                 Vec3, v,
                 Real, f,
                 Real, w,
                 Real, dw)

void collide(Real dt, Real coeff, Real m0, pic_mark &p0, Real m1, pic_mark &p1)
{

}

SP_APP(pic, "PIC width collision ")
{
    static const int num_of_species = 4;
    char name[num_of_species][10] = // charge /mass
            {
                    "D", //D
                    "T", // T
                    "He", // He
                    "e" // e
            };

    Real mass[num_of_species] = // charge /mass
            {
                    2, //D
                    3.0 / 3, // T
                    4.0 / 4, // He
                    1.0 / 1836 // e
            };
    Real charge[num_of_species] = // charge /mass
            {
                    1, //D
                    1.0, // T
                    1.0, // He
                    -1.0 // e
            };

    Real coeff[num_of_species][num_of_species];
    Real T[num_of_species];

    size_t num_of_particle = 1000;
    size_t num_of_step = 10;
    size_t check_point = 1;
    Real dt = 0.2;
    Real time = 1.0;
    std::vector<pic_mark> p[num_of_species];

    std::mt19937 rnd_gen;

    std::cout << std::setw(10) << "Time";
    for (int i = 0; i < num_of_species; ++i)
    {
        p[i].resize(num_of_species);

//        auto p_generator = simple_particle_generator(*ion, mesh->extents(), T[i]);

        for (auto &ps:p[i])
        {

//            ps = p_generator(rnd_gen);
            ps.f = 1.0;
            ps.v = 0.0;
            ps.w = 0;
            ps.dw = 0;
        }
        std::cout << std::setw(10) << name[i] << ",";
    }
    std::cout << std::endl;

    size_t count = 0;
    for (int step = 0; step < num_of_step; ++step)
    {
        for (int i = 0; i < num_of_species; ++i)
        {
            for (int j = i; j < num_of_species; ++j)
            {
                for (auto &pi:p[i])
                    for (auto &pj:p[j])
                    {
                        if (&pi == &pj)continue;

                        collide(dt, coeff[i][j], mass[i], pi, mass[j], pj);
                    }
            }
        }
        std::cout << std::setw(10) << time;
        for (int i = 0; i < num_of_species; ++i)
        {
            Real T = 0.0;
            for (auto &ps:p[i])
            {
                ps.w += ps.dw;
                ps.dw = 0;
                T += ps.f * ps.w * dot(ps.v, ps.v) * 0.5 * mass[i];
            }
            std::cout << std::setw(10) << T << ",";

            if (count % check_point == 0)
            {
                LOGGER << save(name[i], p[i]) << std::endl;
            }
        }
        std::cout << std::endl;


        ++count;
        time += dt;
    }
}