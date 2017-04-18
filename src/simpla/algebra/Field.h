/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/utilities/sp_def.h>
#include <simpla/utilities/EntityId.h>
#include <cstring>  // for memset
#include "Algebra.h"
#include "nTuple.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/CalculusPolicy.h"
#include "simpla/concept/Printable.h"
#include "simpla/data/all.h"
#include "simpla/engine/Attribute.h"
#include "simpla/engine/MeshBlock.h"
#include "simpla/utilities/FancyStream.h"
#include "simpla/utilities/Range.h"
namespace simpla {
namespace mesh {

CHECK_MEMBER_TYPE(check_entity_id, entity_id)
CHECK_MEMBER_TYPE(check_scalar_type, scalar_type)

template <typename TM>
struct mesh_traits {
    typedef TM type;
    typedef typename check_entity_id<TM>::type entity_id;
    typedef typename check_scalar_type<TM>::type scalar_type;

    //    template <int IFORM, int DOF>
    //    struct Shift {
    //        template <typename... Args>
    //        Shift(Args&&... args) {}
    //        constexpr entity_id const& operator()(TM const& m, id const& s) const { return s; }
    //    };
};

}  // namespace mesh{

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
    typedef Array<value_type, NDIMS> sub_array_type;
    static constexpr int num_of_subs = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;
    std::shared_ptr<sub_array_type> m_data_[num_of_subs];
    mesh_type const* m_mesh_;

   public:
    explicit FieldView(mesh_type* m, std::shared_ptr<data::DataEntity> const& d = nullptr)
        : m_mesh_(m), engine::Attribute(m, d){};

    FieldView(mesh_type& m) : m_mesh_(&m), engine::Attribute(&m){};

    template <typename... Args>
    FieldView(mesh_type* m, Args&&... args) : m_mesh_(m), engine::Attribute(m, std::forward<Args>(args)...){};

    template <typename... Args>
    FieldView(mesh_type& m, Args&&... args) : FieldView(&m, std::forward<Args>(args)...){};

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
        for (int i = 0; i < num_of_subs; ++i) {
            if (m_data_[i] != nullptr) { m_data_[i]->Clear(); }
        }
    }
    virtual bool empty() const { return m_data_[0] == nullptr; }

    this_type& operator=(this_type const& other) {
        Assign(other);
        for (int i = 0; i < num_of_subs; ++i) { m_data_[i] = other.m_data_[i]; }
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    }

    void PushData(shared_ptr<data::DataBlock> d) {
        //        m_chart_ = dynamic_cast<mesh_type const*>(engine::Attribute::GetMesh());
        //        ASSERT(m_chart_ != nullptr && m_chart_->GetMeshBlock()->GetGUID() == m->GetGUID());
        //        if (d == nullptr) {
        //            for (int i = 0; i < num_of_subs; ++i) {
        //                m_data_[i] = std::make_shared<sub_array_type>(m->GetInnerIndexBox(), m->GetOuterIndexBox());
        //            }
        //        } else if (d->isArray() && num_of_subs > 1) {
        //            auto& t = d->cast_as<data::DataEntityWrapper<void*>>();
        //            for (int i = 0; i < num_of_subs; ++i) {
        //                m_data_[i] = t.Get(i)->cast_as<data::DataEntityWrapper<sub_array_type>>().get();
        //            }
        //        } else {
        //            m_data_[0] = d->cast_as<data::DataEntityWrapper<sub_array_type>>().get();
        //        }
    }
    virtual std::shared_ptr<data::DataBlock> PopData() {
        //        std::shared_ptr<data::DataEntity> t = nullptr;
        //        if (num_of_subs == 1) {
        //            t = std::make_shared<data::DataEntityWrapper<sub_array_type>>(m_data_[0]);
        //        } else {
        //            auto t_array = std::make_shared<data::DataEntityWrapper<void*>>();
        //            for (int i = 0; i < num_of_subs; ++i) {
        //                auto res = std::make_shared<data::DataEntityWrapper<sub_array_type>>(m_data_[i]);
        //                t_array->Add(res);
        //                m_data_[i].reset();
        //                t = t_array;
        //            }
        //        }
        //        return std::make_pair(m_chart_->GetMeshBlock(), t);
        return nullptr;
    }
    sub_array_type const& operator[](unsigned int i) const { return *m_data_[i % num_of_subs]; }
    sub_array_type& operator[](unsigned int i) { return *m_data_[i % num_of_subs]; }

    value_type const& at(EntityId const& s) const { return (*m_data_[s.w])(s.x, s.y, s.z); }
    value_type& at(EntityId const& s) { return (*m_data_[s.w])(s.x, s.y, s.z); }

   private:
    template <typename... Others>
    decltype(auto) get_(std::integral_constant<bool, true>, unsigned int n, Others&&... others) {
        return m_data_[n]->at(std::forward<Others>(others)...);
    }
    template <typename... Others>
    decltype(auto) get_(std::integral_constant<bool, true>, unsigned int n, Others&&... others) const {
        return m_data_[n]->at(std::forward<Others>(others)...);
    }
    template <typename... Others>
    decltype(auto) get_(std::integral_constant<bool, false>, Others&&... others) {
        return m_data_[0]->at(std::forward<Others>(others)...);
    }
    template <typename... Others>
    decltype(auto) get_(std::integral_constant<bool, false>, Others&&... others) const {
        return m_data_[0]->at(std::forward<Others>(others)...);
    }

   public:
    template <typename... Idx>
    value_type& operator()(index_type i0, Idx&&... idx) {
        return get_(std::integral_constant<bool, (NUMBER_OF_SUB > 1)>(), i0, std::forward<Idx>(idx)...);
    }
    template <typename... Idx>
    value_type const& operator()(index_type i0, Idx&&... idx) const {
        return get_(std::integral_constant<bool, (NUMBER_OF_SUB > 1)>(), i0, std::forward<Idx>(idx)...);
    }

    typedef calculus::template calculator<mesh_type> calculus_policy;

    template <typename... Args>
    decltype(auto) gather(Args&&... args) const {
        return calculus_policy::gather(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) scatter(Args&&... args) {
        return calculus_policy::scatter(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    //    decltype(auto) operator()(point_type const& x) const { return gather(x); }

    //**********************************************************************************************

    template <typename TOP, typename... Args>
    void Foreach_(Range<EntityId> const& r, TOP const& op, Args&&... args) {
        ASSERT(!empty());
        for (int j = 0; j < num_of_subs; ++j) {
            r.foreach ([&](EntityId s) {
                s.w = j;
                op(at(s), calculus_policy::getValue(*m_mesh_, std::forward<Args>(args), s)...);
            });
        }
    }
    template <typename... Args>
    void Foreach(Range<EntityId> const& r, Args&&... args) {
        Foreach_(r, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void Foreach(Args&&... args) {
        SetUp();
        Foreach_(m_mesh_->GetRange(), std::forward<Args>(args)...);
    }
    template <typename Other>
    void Assign(Other const& other) {
        SetUp();
        Foreach_(m_mesh_->GetRange(), tags::_assign(), other);
    }
    template <typename Other>
    void Assign(Range<EntityId> const& r, Other const& other) {
        Foreach_(r, tags::_assign(), other);
    }
};  // class FieldView

// template <typename TM, typename TV, int IFORM, int DOF>
// std::ostream& FieldView<TM, TV, IFORM, DOF>::Print(std::ostream& os, int indent) const {
//    for (int i = 0; i < num_of_subs; ++i) { os << *m_data_[i] << std::endl; }
//    return os;
//}

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
