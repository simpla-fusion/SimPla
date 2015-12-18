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
#include "../../core/io/IO.h"
#include "../../core/io/IOExt.h"

#include "Constants.h"
#include "../../core/physics/PhysicalConstants.h"
#include "../../core/gtl/utilities/config_parser.h"
#include "../../core/gtl/utilities/log.h"
#include "../../core/particle/obsolete/SimpleParticleGenerator.h"


using namespace simpla;

SP_DEFINE_STRUCT(pic_mark, Vec3, v, Real, f, Real, w, Real, dw);

void collide(Real dt, Real coeff, Real m0, pic_mark &p0, Real m1, pic_mark &p1)
{

}

void histogram(int nx, int nbin, Real vmin, Real vmax, std::vector<pic_mark> const &data, std::vector<Real> *res)
{
    res->resize(nbin + 1);

    for (auto &v:*res) { v = 0.0; }

    Real dv = (vmax - vmin) / (nbin);

    for (auto const &p:data)
    {
        if (p.v[nx] < vmin || p.v[nx] > vmax)continue;

        Real v_ = (p.v[nx] - vmin) / dv;
        size_t idx = static_cast<size_t>(v_ + 0.5);
        Real r = v_ - static_cast<Real>(idx);


        (*res)[idx % (nbin + 1)] += p.f * (1 + p.w) * (1.0 - r);
        (*res)[(idx + 1) % (nbin + 1)] += p.f * (1 + p.w) * r;
    }


}

SP_APP(pic, "PIC width collision ")
{
    static const int num_of_species = 5;
    static const int tab_width = 15;
    char name[num_of_species][10] = // charge /mass
            {
                    "H", //H
                    "D", //D
                    "T", // T
                    "He", // He
                    "e" // e
            };

    Real mass[num_of_species] = // charge /mass
            {
                    1, //H
                    2, //D
                    3, // T
                    4, // He
                    1.0 / 1836 // e
            };
    Real charge[num_of_species] = // charge /mass
            {
                    1, //H
                    1, //D
                    1.0, // T
                    1.0, // He
                    -1.0 // e
            };

    Real deltaT[num_of_species] =
            {
                    0.5,
                    0.2,
                    -0.2,
                    -0.5,
                    -0.1
            };

    Real coeff[num_of_species][num_of_species];
    Real T[num_of_species];


    size_t nbin = 50;

    size_t dims[3]{nbin, nbin, nbin};

    size_t num_of_particle = dims[0] * dims[1] * dims[2];
    size_t num_of_step = 1;
    size_t check_point = 1;
    Real dt = 0.2;
    Real time = 1.0;

    Real T0 = 1.0;

    std::vector<pic_mark> p[num_of_species];


    std::vector<Real> hist[3];


    MESSAGE << std::setw(tab_width) << "Time";

    for (int s = 0; s < num_of_species; ++s) { MESSAGE << std::setw(tab_width) << name[s] << ","; }


    nTuple<Real, 3> vmin{-5.0, -5.0, -5.0};

    nTuple<Real, 3> vmax{5.0, 5.0, 5.0};

//    rectangle_distribution<3> box_dist(vmin, vmax);
//
//    std::mt19937 rnd_gen;

    for (int s = 0; s < num_of_species; ++s)
    {


        p[s].resize(num_of_particle);

        Real vT0 = std::sqrt((T0 * 2) / mass[s]);

        Real dv = (vmax[0] - vmin[0]) * (vmax[1] - vmin[1]) * (vmax[2] - vmin[2]) *
                  power3(vT0) / (num_of_particle);

        Real A = dv * std::pow(mass[s] / (2.0 * PI * T0), 1.5);

        Real a = 0.5 * mass[s] / T0;

        Real wA = std::pow(T0 / (deltaT[s] + T0), 1.5);

        Real wa = 0.5 * mass[s] * (-1.0 / T0 + 1.0 / (deltaT[s] + T0));


#pragma omp parallel for
        for (int i = 0; i < dims[0]; ++i)
            for (int j = 0; j < dims[1]; ++j)
                for (int k = 0; k < dims[2]; ++k)
                {
                    auto &ps = p[s][i * dims[1] * dims[2] + j * dims[2] + k];

//                    ps.v = box_dist(rnd_gen) * vT0;

                    ps.v[0] = (vmin[0] + (vmax[0] - vmin[0]) * i / (dims[0] - 1)) * vT0;
                    ps.v[1] = (vmin[1] + (vmax[1] - vmin[1]) * j / (dims[1] - 1)) * vT0;
                    ps.v[2] = (vmin[2] + (vmax[2] - vmin[2]) * k / (dims[2] - 1)) * vT0;

                    ps.f = A * std::exp(-dot(ps.v, ps.v) * a);

                    ps.w = wA * std::exp(-dot(ps.v, ps.v) * wa) - 1.0;

                    ps.dw = 0;

                }

    }
    MESSAGE << std::endl;

    for (int step = 0; step < num_of_step; ++step)
    {
//        for (int si = 0; si < num_of_species; ++si)
//        {
//            for (int sj = si; sj < num_of_species; ++sj)
//            {
//
//#pragma omp parallel for
//                for (size_t i = 0, ie = p[si].size(); i < ie; ++i)
//                {
//                    auto &pi = p[si][i];
//                    for (auto &pj:p[sj])
//                    {
//                        if (&pi == &pj)continue;
//
//                        collide(dt, coeff[si][sj], mass[si], pi, mass[sj], pj);
//                    }
//                }
//            }
//        }
        MESSAGE << std::setw(tab_width) << time;

        for (int si = 0; si < num_of_species; ++si)
        {

            Real T = 0.0;
#pragma omp parallel for reduction(+:T)
            for (size_t i = 0, ie = p[si].size(); i < ie; ++i)
            {
                auto &ps = p[si][i];

                ps.w += ps.dw;

                ps.dw = 0;

                T += ps.f * (1.0 + ps.w) * dot(ps.v, ps.v) * 0.5;
            }

            T *= mass[si] * 2 / 3;
            MESSAGE << std::setw(tab_width) << T << ",";

            if (step % check_point == 0)
            {
                LOGGER << save(name[si], &p[si][0], 3, dims) << std::endl;


            }
        }
        MESSAGE << std::endl;


        time += dt;
    }
}