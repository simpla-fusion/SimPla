/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
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

template <typename>
struct AttributeAdapter : public Object, concept::Printable {};
}  // namespace mesh{

namespace algebra {
namespace calculus {
template <typename...>
class calculator;
}
template <typename, typename, int...>
class FieldView;

// template <typename>
// struct field_traits {
//    static constexpr int iform = VERTEX;
//    static constexpr int dof = 1;
//};
// template <typename TM, typename TV, int IFORM, int DOF>
// struct field_traits<declare::Field_<TM, TV, int_sequence<IFORM, DOF>>> {
//    static constexpr bool is_field = true;
//    static constexpr int iform = IFORM;
//    static constexpr int dof = DOF;
//    typedef TV value_type;
//    typedef TM mesh_type;
//
//   private:
//    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF>> cell_tuple;
//
//   public:
//    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
//                               declare::nTuple_<cell_tuple, 3>>
//        field_value_type;
//};

template <typename TM, typename TV, int IFORM, int DOF>
class FieldView<TM, TV, IFORM, DOF> : public mesh::AttributeAdapter<FieldView<TM, TV, IFORM, DOF>> {
   private:
    typedef FieldView<TM, TV, IFORM, DOF> field_type;
    typedef mesh::AttributeAdapter<FieldView<TM, TV, IFORM, DOF>> attribute_type;
    SP_OBJECT_HEAD(field_type, attribute_type)
   public:
    typedef TV value_type;

    typedef TM mesh_type;
    static constexpr int iform = IFORM;
    static constexpr int dof = DOF;

    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_field;

    typedef typename mesh::mesh_traits<mesh_type>::entity_id entity_id;
    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
                               nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    value_type* m_data_ = nullptr;

    std::shared_ptr<value_type> m_data_holder_ = nullptr;

    mesh_type const* m_mesh_;

   public:
    FieldView() : m_mesh_(nullptr), m_data_(nullptr), m_data_holder_(nullptr){};

    explicit FieldView(this_type const& other)
        : m_data_(const_cast<value_type*>(other.data())),
          m_mesh_(other.mesh()),
          m_data_holder_(other.data_holder()) {}

    //    explicit FieldView(this_type const& other, Shift const& hasher)
    //        : m_data_(const_cast<value_type*>(other.data())),
    //          m_mesh_(other.mesh()),
    //          m_data_holder_(other.data_holder()),
    //          m_shift_(other.m_shift_) {}
    //
    FieldView(mesh_type const* m, value_type* d = nullptr)
        : m_mesh_(m), m_data_(d), m_data_holder_(d, simpla::tags::do_nothing()) {}

    FieldView(std::shared_ptr<mesh_type> const& m, std::shared_ptr<value_type> const& d = nullptr)
        : m_mesh_(m.get()), m_data_(d.get()), m_data_holder_(d) {}

    virtual ~FieldView() {}

    virtual std::ostream& print(std::ostream& os, int indent = 0) const {
        if (m_data_ != nullptr) {
            auto dims = m_mesh_->dimensions();
            size_type s = m_mesh_->size();
            int num_com = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

            if (num_com <= 1) {
                printNdArray(os, m_data_, 3, &dims[0]);

            } else {
                os << "{" << std::endl;
                for (int i = 0; i < num_com; ++i) {
                    os << "[" << i << "] =  ";
                    printNdArray(os, m_data_ + i * s, 3, &dims[0]);
                    os << std::endl;
                }
                os << " }" << std::endl;
            }
        }
        return os;
    }

    this_type& operator=(this_type const& rhs) = delete;

    virtual bool empty() const { return m_data_holder_ == nullptr && m_data_ == nullptr; }

    virtual size_type size() const {
        ASSERT(m_mesh_ != nullptr);
        return m_mesh_->size(IFORM) * DOF;
    }
    auto data_holder() { return m_data_holder_; }
    auto data_holder() const { return m_data_holder_; }
    value_type* data() { return m_data_; }
    value_type const* data() const { return m_data_; }
    mesh_type const* mesh() const { return m_mesh_; }

    virtual void accept(mesh_type const* m, value_type* d) {
        m_mesh_ = m;
        m_data_ = d;
        m_data_holder_ = nullptr;
    }

    virtual void deploy() {
        if (m_data_ == nullptr) {
            if (this->m_data_holder_.get() == nullptr) {
                try {
                    m_data_holder_ = std::shared_ptr<value_type>(new value_type[size()]);
                } catch (std::bad_alloc const&) { THROW_EXCEPTION_BAD_ALLOC(size()); };
            }
            m_data_ = this->m_data_holder_.get();
        }
    };

    virtual void reset() {
        m_data_ = nullptr;
        m_data_holder_.reset();
    };

    virtual void clear() {
        deploy();
        memset(m_data_, 0, size() * sizeof(value_type));
    };
    entity_id const& shift(entity_id const& s) const { return s; }

    virtual void copy(this_type const& other) {
        deploy();
        ASSERT(!other.empty());
        memcpy((void*)(m_data_), (void const*)(other.m_data_), size() * sizeof(value_type));
    };
    decltype(auto) at(entity_id const& s) const {
        ASSERT(m_data_ != nullptr);
        return m_data_[m_mesh_->hash(shift(s))];
    }
    decltype(auto) at(entity_id const& s) {
        ASSERT(m_data_ != nullptr);
        return m_data_[m_mesh_->hash(shift(s))];
    }
    template <typename... TID>
    value_type& at(TID&&... s) {
        return at(m_mesh_->pack(IFORM, DOF, std::forward<TID>(s)...));
    }
    template <typename... TID>
    value_type at(TID&&... s) const {
        ASSERT(m_data_ != nullptr);
        return at(m_mesh_->pack(IFORM, DOF, std::forward<TID>(s)...));
    }

    decltype(auto) operator[](entity_id const& s) const { return at(s); }
    decltype(auto) operator[](entity_id const& s) { return at(s); }
    /** @name as_function  @{*/

    typedef calculus::template calculator<TM> calculus_policy;

    template <typename... Args>
    decltype(auto) gather(Args&&... args) const {
        return calculus_policy::gather(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) scatter(Args&&... args) {
        return calculus_policy::scatter(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    /**@}*/

    template <typename... Args>
    decltype(auto) operator()(index_type s, Args&&... args) const {
        return get(s, std::forward<Args>(args)...);
    }

    decltype(auto) operator()(point_type const& x) const { return gather(x); }

    template <typename... Args>
    void assign(Args&&... args) {
        apply(tags::_assign(), std::forward<Args>(args)...);
    }
    //**********************************************************************************************

    template <typename TOP, typename... Args>
    void apply_(Range<entity_id> const& r, TOP const& op, Args&&... args) {
        int num_com = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);

        for (int i = 0; i < num_com; ++i) {
            for (int j = 0; j < DOF; ++j) {
                r.foreach ([&](entity_id s) {
                    s = shift(s);
                    op(m_data_[m_mesh_->hash(s)],
                       calculus_policy::get_value(*m_mesh_, std::forward<Args>(args), s)...);
                });
            }
        }
    }

    template <typename... Args>
    void apply(Args&&... args) {
        apply_(m_mesh_->range(), std::forward<Args>(args)...);
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
    using typename base_type::value_type;
    using typename base_type::entity_id;
    //    using typename base_type::Shift;
    using base_type::iform;
    using base_type::dof;
    using base_type::at;
    //    using base_type::get;
    using base_type::mesh;
    using base_type::deploy;
    using base_type::print;

    template <int... N>
    this_type operator[](PlaceHolder<N...> const& p) const {
        return this_type(*this, p);
    }
    decltype(auto) operator[](entity_id const& s) const { return at(s); }
    decltype(auto) operator[](entity_id const& s) { return at(s); }
    this_type& operator=(this_type const& rhs) {
        base_type::assign(rhs);
        return *this;
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        base_type::assign(rhs);
        return *this;
    }
};

}  // namespace declare
}  // namespace algebra

template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
