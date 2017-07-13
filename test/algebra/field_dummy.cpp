//
// Created by salmon on 16-12-26.
//

#include "simpla/algebra/algebra.h"
#include "simpla/mesh/DummyMesh.h"

using namespace simpla;

int main(int argc, char **argv) {
    index_box_type i_box = {{0, 0, 0}, {2, 4, 3}};
    box_type x_box{{0, 0, 0}, {1, 2, 3}};
    {
        typedef DummyMesh mesh_type;
        mesh_type m(i_box);

        Field<mesh_type, Real, VERTEX> f(&m);
        Field<mesh_type, Real, VERTEX> g(&m);

        f.SetUndefined();
        g.SetUndefined();
        //        f[0](0, 2, 3) = 1990;

        CHECK(f.Get());

        f[0] = [](index_type x, index_type y, index_type z) -> Real { return x + y + z; };
        //        g = [&](EntityId const &s) { return 1.0; };
        //        f = 1;
        CHECK(f.Get());

        g = 2;
        f = f * 0.2 + g * 2;
        CHECK(f.Get());
        Field<mesh_type, Real, EDGE> h(&m);
        h.SetUndefined();

        //        h = nTuple<Real, 3>{1, 2, 3};
        CHECK(h.Get());

        //        Field<mesh_type, Real, VOLUME, 3> k(&m);
        //        k.SetUndefined();
        //
        //        k = [](EntityId s) {
        //            return nTuple<Real, 3>{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)};
        //        };
        //        CHECK(k.Get());
        //        f = [](point_type const &x) { return x[0]; };
        //        g = [](EntityId s) -> Real { return s.y; };
        //        CHECK(f.data());
        //        CHECK(g.data());
        //    CHECK(f);
        //    Field<mesh_type, Real, EDGE> E(&m);
        //    Field<mesh_type, Real, FACE> B(&m);
        //    Field<mesh_type, Real, VERTEX, 3> d(&m);
        //    Field<mesh_type, Real, VERTEX> rho(&m);
        //    Field<mesh_type, Real, VERTEX, 8> v(&m);
        //    E.Clear();
        //    rho.Clear();

        //    E = [&](point_type const &x) { return x; };
        //    CHECK(E);
        //    rho = codifferential_derivative(E);
        //    E[I] = 2;

        //    B[2][I] = (E[0][-1_i, 1_j] * v[0][I] - E[0][J - 1] * v[0][J - 1] + E[1][I] * v[1][I] -
        //               E[1][I - 1] * v[1][I - 1]) /
        //              v[3][I];
        //    CHECK(E);
        //    diverge(E);
    }
    CHECK("The End");
}