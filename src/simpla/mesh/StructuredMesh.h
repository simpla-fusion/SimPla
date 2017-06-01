//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_STRUCTUREDMESH_H
#define SIMPLA_STRUCTUREDMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/MeshBase.h>
#include <simpla/geometry/Chart.h>
#include <simpla/utilities/Array.h>
namespace simpla {
namespace mesh {
using namespace simpla::data;
using namespace simpla::algebra;

/**
 *  Structured Mesh
 *  - index space and local coordinates have same origin coordinates
 *
 */
class StructuredMesh : public engine::MeshBase {
    SP_OBJECT_HEAD(StructuredMesh, engine::MeshBase)
   public:
    static constexpr unsigned int NDIMS = 3;
    typedef Real scalar_type;

    template <typename... Args>
    explicit StructuredMesh(Args &&... args) : engine::MeshBase(std::forward<Args>(args)...){};

    ~StructuredMesh() override = default;

    SP_DEFAULT_CONSTRUCT(StructuredMesh);

    DECLARE_REGISTER_NAME("StructuredMesh");

    void Update() override;

    index_box_type GetIndexBox(int tag) const override;

    void RegisterRanges(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix) override;

    point_type point(EntityId s) const override;

    point_type local_coordinates(EntityId s, point_type const &r = point_type{0, 0, 0}) const override;

    template <typename V>
    using data_type = data::DataMultiArray<V, NDIMS>;

    template <typename V, int IFORM, int DOF>
    std::shared_ptr<data_type<V>> make_data() const {
        auto gw = GetGhostWidth();

        int num_of_subs = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
        auto res = std::make_shared<data_type<V>>(num_of_subs * DOF);
        for (int i = 0; i < num_of_subs; ++i) {
            auto id_box = GetIndexBox(EntityIdCoder::m_sub_index_to_id_[IFORM][i]);
            std::get<0>(id_box) -= gw;
            std::get<1>(id_box) += gw;
            for (int j = 0; j < DOF; ++j) { Array<V, NDIMS>(id_box).swap((*res)[i * DOF + j]); }
        }
        return res;
    }

    EntityRange make_range(int IFORM) const {
        EntityRange res;
        int num_of_subs = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
        for (int i = 0; i < num_of_subs; ++i) {
            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
            auto id_box = GetIndexBox(tag);
            res.append(std::make_shared<ContinueRange<EntityId>>(id_box, tag));
        }

        return res;
    }

    void InitializeData(Real time_now) override { engine::MeshBase::InitializeData(time_now); }
    void SetBoundaryCondition(Real time_now, Real time_dt) override {
        engine::MeshBase::SetBoundaryCondition(time_now, time_dt);
    }

   protected:
    point_type m_dx_{1, 1, 1};
    point_type m_x0_{0, 0, 0};

   public:
    template <typename M, typename V, typename U, int IFORM, int DOF>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Field<M, U, IFORM, DOF> const &other,
                ENABLE_IF((std::is_base_of<this_type, M>::value))) const {
        static constexpr int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;

        for (int i = 0; i < num_of_sub; ++i) {
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(tag);
                auto &lhs = f[i * DOF + j];
                auto &rhs = other[i * DOF + j];

                if (r.isNull()) {
                    lhs = rhs;
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == (s.w & 0b111)) { lhs.Assign(idx, rhs); }
                    });
                }
            }
        }
    }
    template <typename M, typename V, int IFORM, int DOF, typename... Others>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Expression<Others...> const &expr,
                ENABLE_IF((std::is_base_of<this_type, M>::value))) const {
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;
        for (int i = 0; i < num_of_sub; ++i) {
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(IFORM);

                auto &lhs = f[i * DOF + j];
                auto rhs = calculator<M>::getValue(expr, *dynamic_cast<M const *>(this), tag | (j << 3));
                if (r.isNull()) {
                    lhs = rhs;
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == (s.w & 0b111)) { lhs.Assign(idx, rhs); }
                    });
                }
            }
        }
    }

    template <typename M, typename V, int IFORM, int DOF, typename Other>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Other const &expr,
                ENABLE_IF((std::is_base_of<this_type, M>::value && std::is_arithmetic<Other>::value))) const {
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;
        for (int i = 0; i < num_of_sub; ++i) {
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(IFORM);

                auto &lhs = f[i * DOF + j];
                auto rhs = expr;
                if (r.isNull()) {
                    lhs = rhs;
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == (s.w & 0b111)) { lhs.Assign(idx, rhs); }
                    });
                }
            }
        }
    }
    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_base_of<this_type, M>::value &&
                           std::is_same<std::result_of_t<TFun(EntityId)>, V>::value))) const {
        static constexpr int num_of_sub = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;

        for (int i = 0; i < num_of_sub; ++i)
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(tag);
                int16_t w = static_cast<int16_t>(tag | (j << 3));
                auto &lhs = f[i * DOF + j];

                if (r.isNull()) {
                    lhs = [&](index_tuple const &idx) {
                        EntityId s;
                        s.w = static_cast<int16_t>(w);
                        s.x = static_cast<int16_t>(idx[0]);
                        s.y = static_cast<int16_t>(idx[1]);
                        s.z = static_cast<int16_t>(idx[2]);
                        return fun(s);
                    };
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == s.w) {
                            s.w = w;
                            lhs.Assign(idx, fun(s));
                        }
                    });
                }
            }
    }

    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_base_of<this_type, M>::value &&
                           std::is_same<std::result_of_t<TFun(EntityId)>, nTuple<V, 3>>::value))) const {
        static constexpr int num_of_sub = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;

        for (int i = 0; i < num_of_sub; ++i)
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(tag);
                int16_t w = static_cast<int16_t>(tag | (j << 3));
                auto &lhs = f[i * DOF + j];
                int n = (IFORM == VERTEX || IFORM == VOLUME) ? j : i;

                if (r.isNull()) {
                    lhs = [&](index_tuple const &idx) {
                        EntityId s;
                        s.w = static_cast<int16_t>(w);
                        s.x = static_cast<int16_t>(idx[0]);
                        s.y = static_cast<int16_t>(idx[1]);
                        s.z = static_cast<int16_t>(idx[2]);
                        return fun(s)[n];
                    };
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == s.w) {
                            s.w = w;
                            lhs.Assign(idx, fun(s)[n]);
                        }
                    });
                }
            }
    }
    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(point_type const &)>, nTuple<V, 3>>::value))) const {
        static constexpr int num_of_sub = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;

        for (int i = 0; i < num_of_sub; ++i)
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(tag);
                int16_t w = static_cast<int16_t>(tag | (j << 3));
                auto &lhs = f[i * DOF + j];
                int n = (IFORM == VERTEX || IFORM == VOLUME) ? j : i;
                if (r.isNull()) {
                    f[i * DOF + j] = [&](index_tuple const &idx) {
                        EntityId s;
                        s.w = static_cast<int16_t>(w);
                        s.x = static_cast<int16_t>(idx[0]);
                        s.y = static_cast<int16_t>(idx[1]);
                        s.z = static_cast<int16_t>(idx[2]);
                        return fun(this->point(s))[n];
                    };
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == s.w) {
                            s.w = w;
                            lhs.Assign(idx, fun(this->point(s))[n]);
                        }
                    });
                }
            }
    }
    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(point_type const &)>, V>::value))) const {
        static constexpr int num_of_sub = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;

        for (int i = 0; i < num_of_sub; ++i)
            for (int j = 0; j < DOF; ++j) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                auto id_box = GetIndexBox(tag);
                int16_t w = static_cast<int16_t>(tag | (j << 3));
                auto &lhs = f[i * DOF + j];
                if (r.isNull()) {
                    lhs = [&](index_tuple const &idx) {
                        EntityId s;
                        s.w = static_cast<int16_t>(w);
                        s.x = static_cast<int16_t>(idx[0]);
                        s.y = static_cast<int16_t>(idx[1]);
                        s.z = static_cast<int16_t>(idx[2]);
                        return fun(point(s));
                    };
                } else {
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (tag == s.w) {
                            s.w = w;
                            lhs.Assign(idx, fun(point(s)));
                        }
                    });
                }
            }
    }
};
}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_STRUCTUREDMESH_H
