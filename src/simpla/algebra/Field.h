/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <cstring>  // for memset
#include "Algebra.h"
#include "CalculusPolicy.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/engine/Attribute.h"
#include "simpla/engine/MeshBlock.h"
#include "simpla/utilities/Array.h"
#include "simpla/utilities/EntityId.h"
#include "simpla/utilities/FancyStream.h"
#include "simpla/utilities/Range.h"
#include "simpla/utilities/nTuple.h"
#include "simpla/utilities/sp_def.h"
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
    int NUMBER_OF_SUB = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_field;
    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    typedef Array<value_type, NDIMS> array_type;
    std::vector<array_type> m_data_;
    mesh_type* m_mesh_ = nullptr;
    Range<EntityId> m_range_;

   public:
    template <typename... Args>
    explicit FieldView(mesh_type* m, Args&&... args) : engine::Attribute(std::forward<Args>(args)...), m_mesh_(m) {
        engine::Attribute::RegisterAt(m_mesh_);
    };
    template <typename... Args>
    explicit FieldView(mesh_type& m, Args&&... args) : FieldView(&m, std::forward<Args>(args)...){};

    template <typename... Args>
    explicit FieldView(std::shared_ptr<mesh_type> m, Args&&... args)
        : FieldView(m.get(), std::forward<Args>(args)...){};

    FieldView(this_type const& other)
        : engine::Attribute(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_), m_range_(other.m_range_) {
        engine::Attribute::RegisterAt(m_mesh_);
    }
    FieldView(this_type&& other)
        : engine::Attribute(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_), m_range_(other.m_range_) {
        engine::Attribute::RegisterAt(m_mesh_);
    }

    FieldView(this_type const& other, Range<EntityId> r)
        : engine::Attribute(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_), m_range_(r) {
        engine::Attribute::RegisterAt(m_mesh_);
    }

    ~FieldView() override = default;

    int GetIFORM() const override { return IFORM; };
    int GetDOF() const override { return DOF; };
    std::type_info const& value_type_info() const override { return typeid(value_type); };  //!< value type
    std::shared_ptr<engine::Attribute> GetDescription() const override {
        return std::make_shared<engine::AttributeDesc<TV, IFORM, DOF>>(db());
    };
    void SetUp() override { m_data_.resize(static_cast<size_type>(NUMBER_OF_SUB)); }

    void Clear() {
        SetUp();
        for (int i = 0; i < m_data_.size(); ++i) { m_data_[i].Clear(); }
    }
    bool empty() const override { return m_data_.size() == 0; }

    this_type& operator=(this_type const& other) {
        Assign(other);
        return *this;
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    }

    void Push(std::shared_ptr<data::DataBlock> d) override {
        if (d != nullptr) {
            auto& t = d->cast_as<data::DataMultiArray<value_type, NDIMS>>();
            m_data_.resize(NUMBER_OF_SUB);
            for (int i = 0; i < m_data_.size(); ++i) { array_type(t.GetArray(i)).swap(m_data_[i]); }
        }
    }
    std::shared_ptr<data::DataBlock> Pop() override {
        auto res = std::make_shared<data::DataMultiArray<value_type, NDIMS>>(NUMBER_OF_SUB);
        for (int i = 0; i < m_data_.size(); ++i) { array_type(m_data_[i]).swap(res->GetArray(i)); }
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

    //        decltype(auto) operator()(point_type const& x) const { return gather(x); }

    template <typename Other>
    void Assign(Other const& other) {
        SetUp();
        if (!m_range_.empty()) {
            m_range_.foreach ([&](EntityId s) { at(s) = calculus_policy::getValue(*m_mesh_, other, s); });
        } else {
            int num_of_com = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;
            for (int n = 0; n < num_of_com; ++n) {
                for (int d = 0; d < DOF; ++d) {
                    m_data_[n * DOF + d].Foreach([&](index_tuple const& k, value_type& v) {
                        v = calculus_policy::getValue(std::integral_constant<int, IFORM>(), *m_mesh_, other, k[0], k[1],
                                                      k[2], n, d);
                    });
                }
            }
        }
    }
};  // class FieldView

namespace declare {

template <typename TM, typename TV, int IFORM, int DOF>
class Field_ : public FieldView<TM, TV, IFORM, DOF> {
    typedef Field_<TM, TV, IFORM, DOF> this_type;
    typedef FieldView<TM, TV, IFORM, DOF> base_type;

   public:
    template <typename... Args>
    explicit Field_(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    Field_(this_type const& other) = delete;
    Field_(this_type&& other) = delete;
    ~Field_() {}

    using base_type::operator[];
    using base_type::operator=;
    using base_type::operator();
    this_type operator()(Range<EntityId> r) { return this_type(*this, r); }
};

}  // namespace declare

}  // namespace algebra

template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
