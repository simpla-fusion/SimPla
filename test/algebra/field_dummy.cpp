//
// Created by salmon on 16-12-26.
//

#include "simpla/algebra/CalculusPolicy.h"
#include "simpla/engine/Field.h"
using namespace simpla;
struct DummyMesh : public engine::MeshBase {
   public:
    typedef DummyMesh this_type;
    //    size_type m_dims_[3];
    //    Real m_lower_[3];
    //    Real m_upper_[3];
    point_type m_dx_{1, 1, 1};
    point_type m_x0_{0, 0, 0};
    static constexpr unsigned int NDIMS = 3;

    typedef EntityId entity_id_type;
    template <typename V, int IFORM, int... DOF>
    using data_type = nTuple<Array<V, ZSFC<NDIMS>>, (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3, DOF...>;
    index_box_type m_index_box_;

    typedef DummyMesh mesh_type;
    explicit DummyMesh(index_box_type const &i_box) : m_index_box_(i_box) {}
    ~DummyMesh() = default;

    ZSFC<3> GetSpaceFillingCurve(int tag) const { return ZSFC<3>(m_index_box_); }

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

    engine::MeshBase *GetMesh() { return this; };
    engine::MeshBase const *GetMesh() const { return this; };

    virtual index_box_type GetIndexBox(int tag = VERTEX) const { return m_index_box_; }
    virtual point_type local_coordinates(EntityId s, Real const *r) const { return point_type{0, 0, 0}; };

    template <int IFORM, typename TL, typename TR>
    void Fill(TL &lhs, TR const &rhs) const {
        CalculusPolicy<this_type>::Fill<IFORM>(*this, lhs, rhs);
    }

    template <int IFORM, typename TL, typename... Args>
    decltype(auto) GetEntity(TL &lhs, Args &&... args) const {
        return CalculusPolicy<this_type>::GetEntity<IFORM>(*this, lhs, std::forward<Args>(args)...);
    }

    size_type GetNumberOfEntity(int IFORM = VERTEX) const {
        return calculus::reduction<tags::multiplication>(std::get<1>(m_index_box_) - std::get<0>(m_index_box_)) *
               ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
    }
    bool empty() const { return false; }
};

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