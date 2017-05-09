/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/Attribute.h>
#include <simpla/engine/Domain.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/utilities/Array.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/nTuple.h>
#include <simpla/utilities/sp_def.h>
#include <cstring>  // for memset
#include "Algebra.h"
#include "CalculusPolicy.h"
namespace simpla {

namespace algebra {
namespace calculus {
template <typename...>
class calculator;
}

template <typename TM, typename TV, int IFORM, int DOF>
class FieldView : public engine::Attribute {
   private:
    typedef FieldView<TM, TV, IFORM, DOF> this_type;

   public:
    typedef TV value_type;
    typedef TM mesh_type;

    static constexpr int iform = IFORM;
    static constexpr int dof = DOF;
    static constexpr int NDIMS = mesh_type::NDIMS;
    static constexpr int NUMBER_OF_SUB = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_field;
    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    typedef Array<value_type, NDIMS> array_type;
    array_type m_data_[NUMBER_OF_SUB];
    mesh_type* m_mesh_ = nullptr;
    EntityRange m_range_;

   public:
    template <typename... Args>
    explicit FieldView(engine::Domain* d, Args&&... args)
        : engine::Attribute(IFORM, DOF, typeid(value_type), d,
                            std::make_shared<data::DataTable>(std::forward<Args>(args)...)),
          m_mesh_(dynamic_cast<mesh_type*>(engine::Attribute::GetDomain()->GetMesh())){};
    template <typename... Args>
    explicit FieldView(engine::MeshBase* d, Args&&... args)
        : engine::Attribute(IFORM, DOF, typeid(value_type), d,
                            std::make_shared<data::DataTable>(std::forward<Args>(args)...)),
          m_mesh_(dynamic_cast<mesh_type*>(engine::Attribute::GetDomain()->GetMesh())){};

    FieldView(this_type const& other) : engine::Attribute(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(other.m_data_[i]).swap(m_data_[i]); }
    }

    FieldView(this_type&& other) : engine::Attribute(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(std::move(other.m_data_[i])).swap(m_data_[i]); }
    }

    FieldView(this_type const& other, EntityRange const& r)
        : engine::Attribute(other), m_mesh_(other.m_mesh_), m_range_(r) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(other.m_data_[i]).swap(m_data_[i]); }
    }

    ~FieldView() override = default;

    size_type size() const override { return m_range_.size() * DOF; }

    void Clear() {
        SetUp();
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { m_data_[i].Clear(); }
    }

    bool empty() const override { return m_data_[0].empty(); }

    this_type& operator=(this_type const& other) {
        Assign(other);
        return *this;
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    };

    void Push(const std::shared_ptr<data::DataBlock>& d, EntityRange const& r) override {
        Click();
        m_range_ = r;

        if (d != nullptr) {
            auto& t = d->cast_as<data::DataMultiArray<value_type, NDIMS>>();
            for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(t.GetArray(i)).swap(m_data_[i]); }
            Tag();
        }
    }
    std::shared_ptr<data::DataBlock> Pop() override {
        auto res = std::make_shared<data::DataMultiArray<value_type, NDIMS>>(NUMBER_OF_SUB);
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(m_data_[i]).swap(res->GetArray(i)); }
        return res;
    }

    array_type const& operator[](int i) const { return m_data_[i]; }
    array_type& operator[](int i) { return m_data_[i]; }

    value_type& operator()(index_type i, index_type j = 0, index_type k = 0, index_type w = 0) {
        return m_data_[w](i, j, k);
    }
    value_type const& operator()(index_type i, index_type j = 0, index_type k = 0, index_type w = 0) const {
        return m_data_[w](i, j, k);
    }

    //*****************************************************************************************************************

    typedef calculus::template calculator<mesh_type> calculus_policy;

    value_type const& at(EntityId s) const { return calculus_policy::getValue(*m_mesh_, *this, s); }
    value_type& at(EntityId s) { return calculus_policy::getValue(*m_mesh_, *this, s); }
    value_type const& operator[](EntityId s) const { return at(s); }
    value_type& operator[](EntityId s) { return at(s); }

    template <typename... Args>
    auto gather(Args&&... args) const {
        return calculus_policy::gather(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto scatter(Args&&... args) {
        return calculus_policy::scatter(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    void SetUp() override {
        engine::Attribute::SetUp();
        static constexpr int id_2_sub[4][3] = {{0, 0, 0}, {1, 2, 4}, {6, 5, 3}, {7, 7, 7}};

        auto gw = m_mesh_->GetGhostWidth();

        for (int i = 0; i < NUMBER_OF_SUB; ++i) {
            if (!m_data_[i].empty()) { continue; }
            auto ib_box = m_mesh_->GetIndexBox(id_2_sub[IFORM][(i / DOF) % 3]);
            std::get<0>(ib_box) -= gw;
            std::get<1>(ib_box) += gw;
            array_type(ib_box).swap(m_data_[i]);
        }

        Tag();
    }

    template <typename Other>
    void Assign(Other const& other) {
        SetUp();

        if (m_range_.isNull()) {
            static int tag[4][3] = {{0, 0, 0}, {1, 2, 4}, {6, 5, 3}, {7, 7, 7}};

            for (int i = 0; i < NUMBER_OF_SUB; ++i) {
                int16_t w = static_cast<int16_t>(((i % DOF) << 3) | tag[IFORM][(i / DOF) % 3]);

                m_data_[i].Foreach([&](index_tuple const& idx, value_type& v) {
                    EntityId s;
                    s.w = w;
                    s.x = static_cast<int16_t>(idx[0]);
                    s.y = static_cast<int16_t>(idx[1]);
                    s.z = static_cast<int16_t>(idx[2]);
                    v = calculus_policy::getValue(*m_mesh_, other, s);
                });
            }
        } else if (!m_range_.empty()) {
            auto ib_box = m_mesh_->GetIndexBox();
            for (int i = 0; i < DOF; ++i) {
                m_range_.foreach ([&](EntityId s) {
                    s.w = s.w | static_cast<int16_t>(i << 3);
//                    if (s.x < std::get<0>(ib_box)[0] || s.x >= std::get<1>(ib_box)[0] ||  //
//                        s.y < std::get<0>(ib_box)[1] || s.y >= std::get<1>(ib_box)[1] ||  //
//                        s.z < std::get<0>(ib_box)[2] || s.z >= std::get<1>(ib_box)[2]) {
//                        VERBOSE << s.w << "," << s.x << "," << s.y << "," << s.z << std::endl;
//                    }
                    this->at(s) = calculus_policy::getValue(*m_mesh_, other, s);
                });
            }
        }
    }
    //        static int tag[4][3] = {{0, 0, 0}, {1, 2, 4}, {6, 5, 3}, {7, 7, 7}};
    //        for (int j = 0; j < NUMBER_OF_SUB; ++j) {
    //            VERBOSE << m_data_[j].GetIndexBox() << "~" << m_mesh_->GetIndexBox(tag[IFORM][(j / DOF) % 3]) <<
    //            std::endl;
    //        }
    //        VERBOSE << s.x << "," << s.y << "," << s.z << "   " << std::boolalpha
    //                << m_data_[EntityIdCoder::SubIndex<IFORM, DOF>(s)].empty() << std::endl;
    //        static constexpr int id_2_sub_edge[3] = {1, 2, 4};
    //        static constexpr int id_2_sub_face[3] = {6, 5, 3};
    //        if (m_range_.empty()) {
    //            for (int i = 0; i < NUMBER_OF_SUB; ++i) {
    //                int16_t w = 0;
    //                switch (IFORM) {
    //                    case VERTEX:
    //                        w = static_cast<int16_t>(i << 3);
    //                        break;
    //                    case EDGE:
    //                        w = static_cast<int16_t>(((i % DOF) << 3) | id_2_sub_edge[(i / DOF) % 3]);
    //                        break;
    //                    case FACE:
    //                        w = static_cast<int16_t>(((i % DOF) << 3) | id_2_sub_face[(i / DOF) % 3]);
    //                        break;
    //                    case VOLUME:
    //                        w = static_cast<int16_t>((i << 3) | 0b111);
    //                        break;
    //                    default:
    //                        break;
    //                }
    //                m_data_[i].Foreach([&](index_tuple const& idx, value_type& v) {
    //                    EntityId s;
    //                    s.w = w;
    //                    s.x = static_cast<int16_t>(idx[0]);
    //                    s.y = static_cast<int16_t>(idx[1]);
    //                    s.z = static_cast<int16_t>(idx[2]);
    //                    v = calculus_policy::getValue(*m_mesh_, other, s);
    //                });
    //            }
    //        } else {
    //        }

};  // class FieldView
template <typename TM, typename TV, int IFORM, int DOF>
constexpr int FieldView<TM, TV, IFORM, DOF>::NUMBER_OF_SUB;  //= ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

namespace declare {

template <typename TM, typename TV, int IFORM, int DOF>
class Field_ : public FieldView<TM, TV, IFORM, DOF> {
    typedef Field_<TM, TV, IFORM, DOF> this_type;
    typedef FieldView<TM, TV, IFORM, DOF> base_type;

   public:
    template <typename... Args>
    explicit Field_(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    Field_(this_type const& other) : base_type(other){};
    //    Field_(this_type&& other) = delete;
    ~Field_() {}

    using base_type::operator[];
    using base_type::operator=;
    using base_type::operator();

    this_type operator[](EntityRange const& d) const { return this_type(*this, d); }
};

}  // namespace declare

}  // namespace algebra

template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
