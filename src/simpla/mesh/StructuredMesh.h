//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_STRUCTUREDMESH_H
#define SIMPLA_STRUCTUREDMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/MeshBase.h>
#include <simpla/utilities/Array.h>
namespace simpla {
namespace mesh {
using namespace simpla::data;
using namespace simpla::algebra;
class StructuredMesh : public engine::MeshBase {
    SP_OBJECT_HEAD(StructuredMesh, engine::MeshBase)
   public:
    static constexpr unsigned int NDIMS = 3;
    typedef Real scalar_type;

    explicit StructuredMesh(engine::Domain *d) : engine::MeshBase(d){};
    ~StructuredMesh() override = default;
    SP_DEFAULT_CONSTRUCT(StructuredMesh);
    DECLARE_REGISTER_NAME("StructuredMesh");

    void RegisterRanges(std::map<std::string, EntityRange> &ranges, std::shared_ptr<geometry::GeoObject> const &g,
                        std::string const &prefix) override;

    virtual point_type point(index_type i, index_type j, index_type k) const {
        return point_type{std::fma(i, m_dx_[0], m_x0_[0]), std::fma(j, m_dx_[1], m_x0_[1]),
                          std::fma(k, m_dx_[2], m_x0_[2])};
    }

    point_type point(EntityId s) const override {
        return point(s, point_type{EntityIdCoder::m_id_to_coordinates_shift_[s.w & 7][0],  //
                                   EntityIdCoder::m_id_to_coordinates_shift_[s.w & 7][1],  //
                                   EntityIdCoder::m_id_to_coordinates_shift_[s.w & 7][2]});
    }
    template <typename V>
    using data_type = data::DataMultiArray<V, NDIMS>;

    template <typename V, int IFORM, int DOF>
    std::shared_ptr<data_type<V>> make_data() const {
        auto gw = GetGhostWidth();

        int num_of_subs = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
        auto res = std::make_shared<data::DataMultiArray<V, NDIMS>>(num_of_subs * DOF);
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

    /**
    *\verbatim
    *                ^s (dl)
    *               /
    *   (dz) t     /
    *        ^    /
    *        |  110-------------111
    *        |  /|              /|
    *        | / |             / |
    *        |/  |            /  |
    *       100--|----------101  |
    *        | m |           |   |
    *        |  010----------|--011
    *        |  /            |  /
    *        | /             | /
    *        |/              |/
    *       000-------------001---> r (dr)
    *
    *\endverbatim
    */
    point_type point(EntityId id, point_type const &pr) const override {
        Real r = pr[0], s = pr[1], t = pr[2];

        Real w0 = (1 - r) * (1 - s) * (1 - t);
        Real w1 = r * (1 - s) * (1 - t);
        Real w2 = (1 - r) * s * (1 - t);
        Real w3 = r * s * (1 - t);
        Real w4 = (1 - r) * (1 - s) * t;
        Real w5 = r * (1 - s) * t;
        Real w6 = (1 - r) * s * t;
        Real w7 = r * s * t;
        point_type res{0, 0, 0};
        res = point(id.x /**/, id.y /**/, id.z /**/) * w0 + point(id.x + 1, id.y, id.z) * w1 +
              point(id.x /**/, id.y + 1, id.z /* */) * w2 + point(id.x + 1, id.y + 1, id.z) * w3 +
              point(id.x /**/, id.y /* */, id.z + 1) * w4 + point(id.x + 1, id.y, id.z + 1) * w5 +
              point(id.x /*  */, id.y + 1, id.z + 1) * w6 + point(id.x + 1, id.y + 1, id.z + 1) * w7;

        return res;
    }

    void InitializeData(Real time_now) override {}

    point_type map(point_type const &x) const override {
        return point_type{std::fma(x[0], m_dx_[0], m_x0_[0]), std::fma(x[1], m_dx_[1], m_x0_[1]),
                          std::fma(x[2], m_dx_[2], m_x0_[2])};
    }

    point_type inv_map(point_type const &x) const override {
        return point_type{std::fma(x[0], m_i_dx_[0], m_i_x0_[0]), std::fma(x[1], m_i_dx_[1], m_i_x0_[1]),
                          std::fma(x[2], m_i_dx_[2], m_i_x0_[2])};
    }

    void SetOrigin(point_type x) override { m_x0_ = x; }
    void SetDx(point_type dx) override { m_dx_ = dx; }
    point_type const &GetOrigin() override { return m_x0_; }
    point_type const &GetDx() override { return m_dx_; }
    void SetUp() override {
        m_i_dx_ = 1.0 / m_dx_;
        m_i_x0_ = -m_x0_ / m_dx_;
    }

   private:
    point_type m_dx_{1, 1, 1};
    point_type m_x0_{0, 0, 0};

    point_type m_i_dx_{1, 1, 1};
    point_type m_i_x0_{0, 0, 0};

   public:
    //    typedef calculator<this_type> calculus_policy;

    template <typename M, typename V, int IFORM, int DOF, typename... Others>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Expression<Others...> const &expr,
                ENABLE_IF((std::is_base_of<this_type, M>::value))) const {
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;

        if (r.isNull()) {
            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    f[i * DOF + j] = calculator<M>::getValue(*dynamic_cast<M const *>(this), expr,
                                                             EntityIdCoder::m_sub_index_to_id_[IFORM][i] | (j << 3));
                }
            }
        } else {
            auto id_box = GetIndexBox(IFORM);

            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                    auto lhs = f[i * DOF + j];
                    auto rhs = calculator<M>::getValue(*dynamic_cast<M const *>(this), expr, tag | (j << 3));
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (in_box(id_box, idx) && tag == (s.w & 0b111)) { lhs.Assign(idx, rhs); }
                    });
                }
            }
        }

        //        for (int i = 0; i < NUMBER_OF_SUB; ++i) {
        //            int w = EntityIdCoder::m_sub_index_to_id_[IFORM][i / DOF] | ((i % DOF) << 3);
        //            f[i].Assign(m_range_[i / DOF], calculator<M>::getValue(*this, other, w));
        //        }
    }

    template <typename M, typename V, int IFORM, int DOF>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Field<M, V, IFORM, DOF> const &other,
                ENABLE_IF((std::is_base_of<this_type, M>::value))) const {
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;

        if (r.isNull()) {
            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) { f[i * DOF + j] = other[i * DOF + j]; }
            }
        } else {
            auto id_box = GetIndexBox(IFORM);

            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    auto &lhs = f[i * DOF + j];
                    auto &rhs = other[i * DOF + j];

                    int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (in_box(id_box, idx) && tag == (s.w & 0b111)) { lhs.Assign(idx, rhs); }
                    });
                }
            }
        }
    }

    template <typename M, typename V, int IFORM, int DOF, typename Other>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, Other const &other,
                ENABLE_IF((std::is_base_of<this_type, M>::value &&           //
                           !concept::is_callable<Other(EntityId)>::value &&  //
                           !concept::is_callable<Other(point_type const &)>::value))) const {
        static_assert(std::is_base_of<this_type, M>::value, "illegal mesh type");
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;

        if (r.isNull()) {
            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    f[i * DOF + j] = calculator<M>::getValue(*dynamic_cast<M const *>(this), other,
                                                             EntityIdCoder::m_sub_index_to_id_[IFORM][i] | (j << 3));
                }
            }
        } else {
            auto id_box = GetIndexBox(IFORM);
            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                    auto &lhs = f[i * DOF + j];
                    auto rhs = calculator<M>::getValue(*dynamic_cast<M const *>(this), other, tag | (j << 3));
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (in_box(id_box, idx) && tag == (s.w & 0b111)) { lhs.Assign(idx, rhs); }
                    });
                }
            }
        }
    };

    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_base_of<this_type, M>::value &&
                           std::is_same<std::result_of_t<TFun(EntityId)>, V>::value))) const {
        int num_of_sub = IFORM == VERTEX || IFORM == VOLUME ? 1 : 3;

        if (r.isNull()) {
            for (int i = 0; i < num_of_sub * DOF; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    int w = EntityIdCoder::m_sub_index_to_id_[IFORM][i] | (j << 3);
                    f[i * DOF + j] = [=](index_tuple const &idx) {
                        EntityId s;
                        s.w = static_cast<int16_t>(w);
                        s.x = static_cast<int16_t>(idx[0]);
                        s.y = static_cast<int16_t>(idx[1]);
                        s.z = static_cast<int16_t>(idx[2]);
                        return fun(s);
                    };
                }
            }
        } else {
            auto id_box = GetIndexBox(IFORM);
            for (int i = 0; i < num_of_sub; ++i) {
                for (int j = 0; j < DOF; ++j) {
                    int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
                    auto &lhs = f[i * DOF + j];
                    r.foreach ([&](EntityId s) {
                        index_tuple idx{s.x, s.y, s.z};
                        if (in_box(id_box, idx) && tag == (s.w & 0b111)) {
                            s.w = static_cast<int16_t>((s.w & 0b111) | (j << 3));
                            lhs.Assign(idx, fun(s));
                        }
                    });
                }
            }
        }
    }
    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_base_of<this_type, M>::value &&
                           std::is_same<std::result_of_t<TFun(point_type const &)>, V>::value))) const {
        Assign(f, r, [&](EntityId s) { return fun(this->point(s)); });
    }
    template <typename M, typename V, int IFORM, int DOF, typename TFun>
    void Assign(Field<M, V, IFORM, DOF> &f, EntityRange const &r, TFun const &fun,
                ENABLE_IF((std::is_base_of<this_type, M>::value &&
                           std::is_same<std::result_of_t<TFun(point_type const &)>, nTuple<V, 3>>::value))) const {
        Assign(f, r, [&](EntityId s) {
            return calculator<M>::sample(*dynamic_cast<M const *>(this), s, fun(this->point(s)));
        });
    }
};
}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_STRUCTUREDMESH_H
