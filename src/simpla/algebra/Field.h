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

    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_field;
    typedef typename mesh_type::entity_id entity_id;

    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    std::shared_ptr<value_type> m_data_ = nullptr;

    mesh_type const* m_mesh_;

   public:
    explicit FieldView(mesh_type const* m = nullptr, std::shared_ptr<value_type> const& d = nullptr)
        : m_mesh_(m), m_data_(d){};

    FieldView(this_type const& other) : m_data_(other.m_data_), m_mesh_(other.m_mesh_) {}

    FieldView(this_type&& other) : m_data_(other.m_data_), m_mesh_(other.m_mesh_) {}

    virtual ~FieldView() {}

    virtual std::ostream& print(std::ostream& os, int indent = 0) const {
        if (m_data_ != nullptr) {
            auto dims = m_mesh_->dimensions();
            size_type s = m_mesh_->size();
            int num_com = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

            if (num_com <= 1) {
                printNdArray(os, m_data_.get(), 3, &dims[0]);

            } else {
                os << "{" << std::endl;
                for (int i = 0; i < num_com; ++i) {
                    os << "[" << i << "] =  ";
                    printNdArray(os, m_data_.get() + i * s, 3, &dims[0]);
                    os << std::endl;
                }
                os << " }" << std::endl;
            }
        }
        return os;
    }

    this_type& operator=(this_type const& rhs) {
        copy(rhs);
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR const& rhs) {
        assign(rhs);
        return *this;
    }
    bool empty() const { return m_data_ == nullptr; }

    size_type size() const { return m_mesh_->size(IFORM) * DOF; }

    virtual std::shared_ptr<value_type> data() { return m_data_; }
    virtual std::shared_ptr<value_type> data() const { return m_data_; }
    virtual mesh_type const* mesh() const { return m_mesh_; }

    virtual void deploy() {
        m_data_ = data();
        m_mesh_ = mesh();
        ASSERT(m_mesh_ != nullptr);

        if (m_data_ == nullptr) {
            if (this->m_data_ == nullptr) {
                try {
                    m_data_ = std::shared_ptr<value_type>(new value_type[size()]);
                } catch (std::bad_alloc const&) { THROW_EXCEPTION_BAD_ALLOC(size()); };
            }
        }
        ASSERT(m_data_ != nullptr);
    };

    void reset() { m_data_.reset(); };

    void clear() {
        deploy();
        memset(m_data_.get(), 0, size() * sizeof(value_type));
    };

    void copy(this_type const& other) {
        deploy();
        if (!other.empty()) {
            memcpy((void*)(m_data_), (void const*)(other.m_data_.get()), size() * sizeof(value_type));
        };
    };

    entity_id const& shift(entity_id const& s) const { return s; }

    decltype(auto) at(entity_id const& s) const { return m_data_.get()[m_mesh_->hash(shift(s))]; }

    decltype(auto) at(entity_id const& s) { return m_data_.get()[m_mesh_->hash(shift(s))]; }

    template <typename... TID>
    decltype(auto) at(TID&&... s) {
        return m_data_.get()[(m_mesh_->hash(std::forward<TID>(s)...))];
    }
    template <typename... TID>
    decltype(auto) at(TID&&... s) const {
        return m_data_.get()[(m_mesh_->hash(std::forward<TID>(s)...))];
    }
    template <typename... TID>
    decltype(auto) operator()(index_type i0, TID&&... s) {
        return at(i0, std::forward<TID>(s)...);
    }

    template <typename... Args>
    decltype(auto) operator()(index_type i0, Args&&... args) const {
        return at(i0, std::forward<Args>(args)...);
    }

    decltype(auto) operator[](entity_id const& s) const { return at(s); }

    decltype(auto) operator[](entity_id const& s) { return at(s); }

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
    void apply_(Range<entity_id> const& r, TOP const& op, Args&&... args) {
        ASSERT(!empty());
        for (int j = 0; j < DOF; ++j) {
            r.foreach ([&](entity_id s) {
                s.w = j;
                op(at(s), calculus_policy::get_value(*m_mesh_, std::forward<Args>(args), s)...);
            });
        }
    }
    template <typename... Args>
    void apply(Range<entity_id> const& r, Args&&... args) {
        apply_(r, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void apply(Args&&... args) {
        apply_(m_mesh_->range(), std::forward<Args>(args)...);
    }
    template <typename Other>
    void assign(Other const& other) {
        apply_(m_mesh_->range(), tags::_assign(), other);
    }
    template <typename Other>
    void assign(Range<entity_id> const& r, Other const& other) {
        apply_(r, tags::_assign(), other);
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
