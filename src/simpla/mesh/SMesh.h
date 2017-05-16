//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::engine;

struct SMesh : public StructuredMesh {
   public:
    SP_OBJECT_HEAD(SMesh, StructuredMesh)

    explicit SMesh(Domain *d) : StructuredMesh(d){};
    ~SMesh() override = default;

    SP_DEFAULT_CONSTRUCT(SMesh)
    DECLARE_REGISTER_NAME("SMesh");
    void InitializeData(Real time_now) override;

    Field<this_type, Real, VERTEX, 3> m_coordinates_{this, "COORDINATES"_, "name"_ = "Coordinates"};
    Field<this_type, Real, VERTEX, 3> m_vertices_{this /* , "name"_ = "vertices"*/};
    Field<this_type, Real, VOLUME, 9> m_volume_{this /* ,"name"_ = "volume"*/};
    Field<this_type, Real, VOLUME, 9> m_dual_volume_{this /* ,"name"_ = "dual_volume"*/};
    Field<this_type, Real, VOLUME, 9> m_inv_volume_{this /* , "name"_ = "inv_volume"*/};
    Field<this_type, Real, VOLUME, 9> m_inv_dual_volume_{this /* ,"name"_ = "inv_dual_volume"*/};

    point_type point(EntityId s) const override { return StructuredMesh::point(s); }

    Real volume(EntityId s) const override { return m_volume_[s]; }
    Real dual_volume(EntityId s) const override { return m_volume_[s]; }
    Real inv_volume(EntityId s) const override { return m_volume_[s]; }
    Real inv_dual_volume(EntityId s) const override { return m_volume_[s]; }

    typedef calculator<this_type> calculus_policy;
    template <typename M, typename V, int IFORM, int DOF, typename Other>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Other const &other) const {
        static_assert(std::is_base_of<this_type, M>::value, "illegal mesh type");
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;

        for (int i = 0; i < num_of_sub; ++i) {
            for (int j = 0; j < DOF; ++j) {
                auto &lhs = f[i][j];
                auto rhs =
                    calculus_policy::getValue(*this, other, EntityIdCoder::m_sub_index_to_id_[IFORM][i] | (j << 3));
                r[i]->foreach ([&](EntityId s) {
                        lhs[index_tuple{s.x,s.y,s.z}]=
                });
            }
        }
    };

    //    template <typename V, int IFORM, int DOF, typename Other>
    //    void Assign(Field<this_type, V, IFORM, DOF> &f, EntityRange const &r, Other const &other,
    //                ENABLE_IF(!(concept::is_callable<Other(EntityId)>::value ||
    //                            concept::is_callable<Other(point_type const &)>::value))) {
    //        DoSetUp();
    //
    //        for (int i = 0; i < NUMBER_OF_SUB; ++i) {
    //            int w = EntityIdCoder::m_sub_index_to_id_[IFORM][i / DOF] | ((i % DOF) << 3);
    //            f[i].Assign(m_range_[i / DOF], calculus_policy::getValue(*this, other, w));
    //        }
    //    }
    //    template <typename V, int IFORM, int DOF, typename TFun>
    //    void Assign(Field<this_type, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
    //                ENABLE_IF((std::is_same<std::result_of_t<TFun(EntityId)>, V>::value))) {
    //        DoSetUp();
    //
    //        for (int i = 0; i < NUMBER_OF_SUB; ++i) {
    //            int w = EntityIdCoder::m_sub_index_to_id_[IFORM][i / DOF] | ((i % DOF) << 3);
    //            f[i].Assign(m_range_[i / DOF], [&](EntityId s) {
    //                s.w = static_cast<int16_t>(w);
    //                return fun(s);
    //            });
    //        }
    //    }
    //    template <typename V, int IFORM, int DOF, typename TFun>
    //    void Assign(Field<this_type, V, IFORM, DOF> &f, TFun const &fun,
    //                ENABLE_IF((std::is_same<std::result_of_t<TFun(point_type const &)>, V>::value))) {
    //        Assign([&](EntityId s) { return fun(this->point(s)); });
    //    }
    //    template <typename V, int IFORM, int DOF, typename TFun>
    //    void Assign(Field<this_type, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
    //                ENABLE_IF(((!std::is_same<field_value_type, V>::value) &&
    //                           std::is_same<std::result_of_t<TFun(point_type const &)>, field_value_type>::value))) {
    //        Assign([&](EntityId s) { return calculus_policy::sample(*this, s, fun(this->point(s))); });
    //    }
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
