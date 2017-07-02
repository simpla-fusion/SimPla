//
// Created by salmon on 16-12-26.
//

#include "simpla/algebra/all.h"

using namespace simpla;
struct DummyMesh {
   public:
    //    size_type m_dims_[3];
    //    Real m_lower_[3];
    //    Real m_upper_[3];
    point_type m_dx_{1, 1, 1};
    point_type m_x0_{0, 0, 0};
    static constexpr unsigned int NDIMS = 3;

    typedef EntityId entity_id_type;
    template <typename V>
    using array_type = Array<V,  ZSFC<NDIMS>>;

    typedef DummyMesh mesh_type;

    explicit DummyMesh() {}
    ~DummyMesh() = default;

    template <typename U, int... N>
    void UpdateArray(nTuple<array_type<U>, N...> &d) const {};

    template <typename TFun>
    void Foreach(TFun const &fun, size_type iform = VERTEX, size_type dof = 1) const {}
    Real volume(EntityId s) const { return 1.0; }
    Real dual_volume(EntityId s) const { return 1.0; }
    Real inv_volume(EntityId s) const { return 1.0; }
    Real inv_dual_volume(EntityId s) const { return 1.0; }

    point_type point(EntityId s) const {
        return point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)};
    }
    point_type point(EntityId s, point_type const &pr) const {
        return point_type{static_cast<Real>(s.x) + pr[0], static_cast<Real>(s.y) + pr[1],
                          static_cast<Real>(s.z) + pr[2]};
    };

    point_type map(point_type const &x) const { return x; }
    point_type inv_map(point_type const &x) const { return x; }
    void SetOrigin(point_type x) { m_x0_ = x; }
    void SetDx(point_type dx) { m_dx_ = dx; }
    point_type const &GetOrigin() { return m_x0_; }
    point_type const &GetDx() { return m_dx_; };
};

int main(int argc, char **argv) {
    index_box_type i_box = {{0, 0, 0}, {2, 4, 3}};
    box_type x_box{{0, 0, 0}, {1, 2, 3}};
    typedef DummyMesh mesh_type;
    DummyMesh m;
    Field<DummyMesh, Real, VERTEX> f(&m);
    Field<DummyMesh, Real, VERTEX> g(&m);

    f.Clear();
    g.Clear();

    f[0](0, 2, 3) = 1990;
    f = 1;
    g = 2;

    f = f + g;
    f = -g * 0.2;

    //    f = [&](point_type const &x) { return x[1] + x[2]; };
    //    g = [&](EntityId const &s) { return 1.0; };

    //    CHECK(f);
    Field<mesh_type, Real, EDGE> E(&m);
    Field<mesh_type, Real, FACE> B(&m);
    Field<mesh_type, Real, VERTEX, 3> d(&m);
    Field<mesh_type, Real, VERTEX> rho(&m);
    Field<mesh_type, Real, VERTEX, 8> v(&m);
    E.Clear();
    rho.Clear();

    //    E = [&](point_type const &x) { return x; };
    //    CHECK(E);
    rho = codifferential_derivative(E);
    //    E[I] = 2;

    //    B[2][I] = (E[0][-1_i, 1_j] * v[0][I] - E[0][J - 1] * v[0][J - 1] + E[1][I] * v[1][I] -
    //               E[1][I - 1] * v[1][I - 1]) /
    //              v[3][I];
    //    CHECK(E);
    //    diverge(E);
}