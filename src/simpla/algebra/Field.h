/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/CalculusPolicy.h>
#include <simpla/concept/Printable.h>
#include <simpla/data/DataBlock.h>
#include <simpla/data/all.h>
#include <simpla/engine/Attribute.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/sp_def.h>
#include <cstring>  // for memset
#include "Algebra.h"
#include "Array.h"
#include "nTuple.h"
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
    static constexpr int NDIMS = 3;
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
    mesh_type const* m_mesh_;

   public:
    FieldView(mesh_type* m, std::shared_ptr<data::DataTable> const& d = nullptr)
        : m_mesh_(m), engine::Attribute(m, d){};

    FieldView(mesh_type& m) : m_mesh_(&m), engine::Attribute(&m){};

    template <typename... Args>
    FieldView(mesh_type* m, Args&&... args) : m_mesh_(m), engine::Attribute(m, std::forward<Args>(args)...){};

    template <typename... Args>
    FieldView(mesh_type& m, Args&&... args) : m_mesh_(&m), engine::Attribute(&m, std::forward<Args>(args)...){};
    FieldView(this_type const& other) = delete;
    FieldView(this_type&& other) = delete;
    virtual ~FieldView() {}

    //    virtual this_type* Clone() const { return new this_type(*this); };
    //    virtual std::shared_ptr<engine::Attribute> GetDescription() const {
    //        return std::make_shared<engine::AttributeDesc<TV, IFORM, DOF>>(db());
    //    };

    virtual int GetIFORM() const { return IFORM; };
    virtual int GetDOF() const { return DOF; };
    virtual std::type_info const& value_type_info() const { return typeid(value_type); };  //!< value type

    virtual void SetUp() {}
    virtual void Initialize() {}
    virtual void Clear() {
        SetUp();
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { m_data_[i].Clear(); }
    }
    virtual bool empty() const { return m_data_[0].empty(); }

    this_type& operator=(this_type const& other) {
        Assign(other);
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    }

    void Push(std::shared_ptr<data::DataBlock> d) {
        if (d != nullptr) {
            auto& t = d->cast_as<data::DataMultiArray<value_type, NDIMS>>();
            for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(t.GetArray(i)).swap(m_data_[i]); }
        }
    }
    std::shared_ptr<data::DataBlock> Pop() {
        auto res = std::make_shared<data::DataMultiArray<value_type, NDIMS>>(NUMBER_OF_SUB);
        for (int i = 0; i < NUMBER_OF_SUB; ++i) { array_type(m_data_[i]).swap(res->GetArray(i)); }
        return res;
    }

    array_type const& operator[](int i) const { return m_data_[i % NUMBER_OF_SUB]; }
    array_type& operator[](int i) { return m_data_[i % NUMBER_OF_SUB]; }

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

    template <typename Other>
    void Assign(Range<EntityId> const& r, Other const& other) {
        SetUp();
        r.foreach ([&](EntityId s) { at(s) = calculus_policy::getValue(*m_mesh_, other, s); });
    }
    //    template <typename TOP, typename... Args>
    //    void Foreach_(Range<EntityId> const& r, TOP const& op, Args&&... args) {
    //        ASSERT(!empty());
    //        SetUp();
    //        r.foreach ([&](EntityId s) { op(at(s), calculus_policy::getValue(*m_block_, std::forward<Args>(args),
    //        s)...); });
    //    }
    //    template <typename... Args>
    //    void Foreach(Range<EntityId> const& r, Args&&... args) {
    //        Foreach_(r, std::forward<Args>(args)...);
    //    }
    //    template <typename... Args>
    //    void Foreach(Args&&... args) {
    //        SetUp();
    //        Foreach_(m_block_->GetRange(GetIFORM()), std::forward<Args>(args)...);
    //    }
    //    template <typename Other>
    //    void Assign(Range<EntityId> const& r, Other const& other) {
    //        Foreach_(r, tags::_assign(), other);
    //    }
};  // class FieldView
template <typename TM, typename TV, int IFORM, int DOF>
constexpr int FieldView<TM, TV, IFORM, DOF>::NUMBER_OF_SUB;

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
};

}  // namespace declare

}  // namespace algebra

template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
