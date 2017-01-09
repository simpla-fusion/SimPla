/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/toolbox/FancyStream.h>
#include <simpla/toolbox/sp_def.h>
#include <cstring>  // for memset
#include "Algebra.h"
#include "nTuple.h"

namespace simpla {
namespace algebra {
namespace declare {

template <typename TM, typename TV, size_type IFORM, size_type DOF>
class Field_;

}  // namespace declare {

template <typename TM, typename TV, size_type IFORM, size_type DOF>
class FieldView;

namespace traits {

//***********************************************************************************************************************
template <typename>
struct mesh_type {
    typedef void type;
};

template <typename TM, typename TV, size_type... I>
struct mesh_type<declare::Field_<TM, TV, I...>> {
    typedef TM type;
};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TM, TV, IFORM, DOF>> : public std::integral_constant<bool, true> {};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TM, TV, IFORM, DOF>> {
    typedef declare::Field_<TM, TV, IFORM, DOF> const& type;
};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct reference<const declare::Field_<TM, TV, IFORM, DOF>> {
    typedef declare::Field_<TM, TV, IFORM, DOF> const& type;
};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct value_type<declare::Field_<TM, TV, IFORM, DOF>> {
    typedef TV type;
};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct rank<declare::Field_<TM, TV, IFORM, DOF>> : public index_const<3> {};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct iform<declare::Field_<TM, TV, IFORM, DOF>> : public index_const<IFORM> {};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct dof<declare::Field_<TM, TV, IFORM, DOF>> : public index_const<DOF> {};

template <typename T>
struct field_value_type {
    typedef T type;
};

template <typename T>
using field_value_t = typename field_value_type<T>::type;

template <typename TM, typename TV, size_type IFORM, size_type DOF>
struct field_value_type<declare::Field_<TM, TV, IFORM, DOF>> {
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
                               declare::nTuple_<cell_tuple, 3>>
        type;
};

}  // namespace traits{

struct IndexShifting {
    index_type i, j, k;
    IndexShifting operator,(IndexShifting const& l) const { return IndexShifting(); }
};

template <typename TM, typename TV, size_type IFORM, size_type DOF>
class FieldView : public concept::Printable {
   private:
    typedef FieldView<TM, TV, IFORM, DOF> this_type;
    //    typedef TM::attribute<TV, IFORM, DOF> base_type;

   public:
    typedef TV value_type;
    typedef TM mesh_type;
    //    static constexpr int NUM_OF_ENTITIES_IN_CELL = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 :
    //    3);
    //    static constexpr NDIMS = traits::num_of_dimension<TM>::value;

   private:
    value_type* m_data_ = nullptr;
    std::shared_ptr<value_type> m_data_holder_ = nullptr;

    mesh_type const* m_mesh_;
    IndexShifting m_shifting_;
    //    ArrayView<value_type, NDIMS> m_view_[(IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3][DOF];

   public:
    FieldView() : m_mesh_(nullptr), m_data_(nullptr){};

    FieldView(FieldView const& other) {}

    FieldView(FieldView const& other, IndexShifting const& s) : FieldView(other) {
        m_shifting_ += s;
    }
    //    template <typename... Args>
    //    explicit FieldView(Args&&... args)
    //        : m_mesh_(nullptr),
    //          m_data_(nullptr),
    //          m_data_holder_(nullptr)
    //    //            : base_type(std::forward<Args>(args)...), m_mesh_(nullptr), m_data_(nullptr)
    //    {}

    FieldView(mesh_type const* m, value_type* d = nullptr)
        : m_mesh_(m), m_data_(d), m_data_holder_(d, simpla::tags::do_nothing()) {}

    FieldView(mesh_type const* m, std::shared_ptr<value_type> const& d)
        : m_mesh_(m), m_data_(d.get()), m_data_holder_(d) {}

    virtual ~FieldView() {}

//    FieldView(this_type const& other) = delete;

//    FieldView(this_type&& other) = delete;

    virtual std::ostream& print(std::ostream& os, int indent = 0) const {
        if (m_data_ != nullptr) {
            auto dims = m_mesh_->dimensions();
            printNdArray(os, m_data_, 3, &dims[0]);
        }
        return os;
    }

    this_type& operator=(this_type const& rhs) = delete;

    virtual mesh_type const& mesh() const { return *m_mesh_; }

    virtual bool empty() const { return m_data_holder_ == nullptr && m_data_ == nullptr; }

    virtual size_type size() const {
        ASSERT(m_mesh_ != nullptr);
        return m_mesh_->size(IFORM, DOF);
    }

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
                } catch (std::bad_alloc const&) {
                    CHECK(size());
                    THROW_EXCEPTION_BAD_ALLOC(size());
                };
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

    virtual void copy(this_type const& other) {
        deploy();
        ASSERT(!other.empty());
        memcpy((void*)(m_data_), (void const*)(other.m_data_), size() * sizeof(value_type));
    };

    template <typename... TID>
    value_type& at(TID&&... s) {
        ASSERT(m_data_ != nullptr);
        return m_data_[m_mesh_->hash(IFORM, DOF, std::forward<TID>(s)...)];
    }
    template <typename... TID>
    value_type at(TID&&... s) const {
        ASSERT(m_data_ != nullptr);
        return m_data_[m_mesh_->hash(IFORM, DOF, std::forward<TID>(s)...)];
    }

    template <typename... Args>
    decltype(auto) get(index_type s, Args&&... args) {
        return m_data_[m_mesh_->hash(IFORM, DOF, s, std::forward<Args>(args)...)];
    }

    template <typename... Args>
    decltype(auto) get(index_type s, Args&&... args) const {
        return m_data_[m_mesh_->hash(IFORM, DOF, s, std::forward<Args>(args)...)];
    }
    template <typename TID>
    decltype(auto) operator[](TID const& s) {
        return at(s);
    }
    template <typename TID>
    decltype(auto) operator[](TID const& s) const {
        return at(s);
    }
};  // class FieldView

namespace declare {

template <typename TM, typename TV, size_type IFORM, size_type DOF>
class Field_ : public FieldView<TM, TV, IFORM, DOF> {
    typedef Field_<TM, TV, IFORM, DOF> this_type;
    typedef FieldView<TM, TV, IFORM, DOF> base_type;

   public:
    typedef TV value_type;
    typedef calculus::template calculator<TM> calculus_policy;

    template <typename... Args>
    explicit Field_(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    ~Field_() {}

    using base_type::at;
    using base_type::get;
    using base_type::mesh;
    using base_type::deploy;
    using base_type::print;

    this_type operator[](int n) const { return this_type(*this, n); }

    this_type operator[](IndexShifting const& s) const { return this_type(*this, s); }

    this_type& operator=(this_type const& rhs) {
        assign(rhs);
        return *this;
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        assign(rhs);
        return *this;
    }

    /** @name as_function  @{*/
    template <typename... Args>
    decltype(auto) gather(Args&&... args) const {
        return calculus_policy::gather(mesh(), *this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) scatter(Args&&... args) {
        return calculus_policy::scatter(mesh(), *this, std::forward<Args>(args)...);
    }

    /**@}*/

    template <typename... Args>
    decltype(auto) operator()(index_type s, Args&&... args) const {
        return get(s, std::forward<Args>(args)...);
    }

    decltype(auto) operator()(point_type const& x) const { return gather(x); }

    template <typename... Args>
    this_type& assign(Args&&... args) {
        calculus_policy::assign(mesh(), *this, std::forward<Args>(args)...);
        return *this;
    }

    template <typename... Args>
    this_type& apply(Args&&... args) {
        deploy();
        calculus_policy::apply(mesh(), *this, std::forward<Args>(args)...);
        return *this;
    }
};

}  // namespace declare
}  // namespace algebra

template <typename TM, typename TV, size_type IFORM = VERTEX, size_type DOF = 1>
using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
