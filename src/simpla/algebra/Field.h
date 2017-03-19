/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/engine/SPObject.h>
#include <simpla/mpl/Range.h>
#include <simpla/toolbox/FancyStream.h>
#include <simpla/toolbox/sp_def.h>
#include <cstring>  // for memset
#include "Algebra.h"
#include "nTuple.h"

namespace simpla {
namespace mesh {

CHECK_TYPE_MEMBER(check_entity_id, entity_id)
CHECK_TYPE_MEMBER(check_scalar_type, scalar_type)

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
template <typename, typename, int...>
class FieldView;
template <typename TM, typename TV, int IFORM, int DOF>
class FieldView<TM, const TV, IFORM, DOF> {};
template <typename TM, typename TV, int IFORM, int DOF>
class FieldView<TM, TV, IFORM, DOF> {
   private:
    typedef FieldView<TM, TV, IFORM, DOF> this_type;

   public:
    typedef TV value_type;

    typedef TM mesh_type;
    static constexpr int iform = IFORM;
    static constexpr int dof = DOF;
    static constexpr int NDIMS = mesh_type::NDIMS;
    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_field;
    typedef typename mesh_type::entity_id entity_id;

    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    mesh_type const* m_mesh_;
    typedef Array<value_type, NDIMS> sub_array_type;
    static constexpr int num_of_subs = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;
    std::shared_ptr<sub_array_type> m_data_[num_of_subs];

   public:
    explicit FieldView(mesh_type const* m, std::shared_ptr<sub_array_type> const* d = nullptr) : m_mesh_(m) {
        SetData(d);
    };

    FieldView(this_type const& other) = delete;
    FieldView(this_type&& other) = delete;

    virtual ~FieldView() {}
    virtual void Initialize() {}
    virtual void Clear() { Update(); }

    sub_array_type const& operator[](int i) const { return *m_data_[i % num_of_subs]; }
    sub_array_type& operator[](int i) { return *m_data_[i % num_of_subs]; }

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        if (m_data_ != nullptr) {
            auto dims = m_mesh_->dimensions();
            size_type s = m_mesh_->size();
            //            int num_com = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

            if (num_of_subs <= 1) {
                printNdArray(os, m_data_, 3, &dims[0]);

            } else {
                os << "{" << std::endl;
                for (int i = 0; i < num_of_subs; ++i) {
                    os << "[" << i << "] =  ";
                    printNdArray(os, m_data_ + i * s, 3, &dims[0]);
                    os << std::endl;
                }
                os << " }" << std::endl;
            }
        }
        return os;
    }

    this_type& operator=(this_type const& rhs) {
        SetData(nullptr);
        Update();
        for (int i = 0; i < num_of_subs; ++i) {
            if (m_data_[i] == nullptr) { m_data_[i]->Copy(rhs.m_data_[i]); }
        }
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    }
    bool empty() const { return m_data_[0] == nullptr; }

    void Update() {
        for (int i = 0; i < num_of_subs; ++i) {
            if (m_data_[i] == nullptr) {
                m_data_[i] == std::make_shared<sub_array_type>(m_mesh_->GetMeshBlock()->GetOuterIndexBox(IFORM, i));
            }
        }
    }
    void SetData(std::shared_ptr<sub_array_type> const* d = nullptr) {
        for (int i = 0; i < num_of_subs; ++i) { m_data_[i] = (d != nullptr) ? d[i] : nullptr; }
    }
    std::shared_ptr<sub_array_type> const* GetData() const { return m_data_; }

    value_type const& at(entity_id const& s) const { return (*m_data_[mesh_type::GetSub(s)])[mesh_type::UnpackIdx(s)]; }

    value_type& at(entity_id const& s) { return (*m_data_[mesh_type::GetSub(s)])[mesh_type::UnpackIdx(s)]; }

    typedef calculus::template calculator<TM> calculus_policy;

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
    void Apply_(Range<entity_id> const& r, TOP const& op, Args&&... args) {
        ASSERT(!empty());
        for (int j = 0; j < DOF; ++j) {
            r.foreach ([&](entity_id s) {
                s.w = j;
                op(at(s), calculus_policy::getValue(*m_mesh_, std::forward<Args>(args), s)...);
            });
        }
    }
    template <typename... Args>
    void Apply(Range<entity_id> const& r, Args&&... args) {
        Apply_(r, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void Apply(Args&&... args) {
        Update();
        Apply_(m_mesh_->range(), std::forward<Args>(args)...);
    }
    template <typename Other>
    void Assign(Other const& other) {
        Update();
        //        Apply_(m_mesh_->range(), tags::_assign(), other);
    }
    template <typename Other>
    void Assign(Range<entity_id> const& r, Other const& other) {
        Apply_(r, tags::_assign(), other);
    }
};  // class FieldView
template <int...>
struct PlaceHolder {};
namespace declare {

template <typename TM, typename TV, int IFORM, int DOF>
class Field_ : public FieldView<TM, TV, IFORM, DOF> {
    typedef Field_<TM, TV, IFORM, DOF> this_type;
    typedef FieldView<TM, TV, IFORM, DOF> base_type;

   public:
    template <typename... Args>
    explicit Field_(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    ~Field_() {}

    Field_(this_type const& other) : base_type(other){};
    Field_(this_type&& other) : base_type(other){};

    using base_type::operator[];
    using base_type::operator=;

    //    template <int... N>
    //    Field_<TM, const TV, IFORM, DOF> operator[](PlaceHolder<N...> const& p) const {
    //        return Field_<TM, const TV, IFORM, DOF>(*this, p);
    //    }
    //    decltype(auto) operator[](entity_id const& s) const { return at(s); }
    //    decltype(auto) operator[](entity_id const& s) { return at(s); }
};

}  // namespace declare
}  // namespace algebra

template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
