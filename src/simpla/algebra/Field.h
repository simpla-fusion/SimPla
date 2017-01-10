/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/mesh/EntityId.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mpl/Range.h>
#include <simpla/toolbox/FancyStream.h>
#include <simpla/toolbox/sp_def.h>
#include <cstring>  // for memset
#include "Algebra.h"
#include "nTuple.h"

namespace simpla {
namespace algebra {
namespace declare {
template <typename TM, typename TV, int...>
class Field_;
}  // namespace declare {

template <typename TM, typename TV, int...>
class FieldView;

template <typename TM>
struct mesh_traits {
    typedef TM type;
    typedef mesh::MeshEntityId id;
};

template <typename TM, typename TV, int... I>
struct mesh_traits<declare::Field_<TM, TV, I...>> {
    typedef TM type;
    typedef mesh::MeshEntityId id;
};

namespace traits {

//***********************************************************************************************************************

template <typename TM, typename TV, int... I>
struct is_field<declare::Field_<TM, TV, I...>> : public std::integral_constant<bool, true> {};

template <typename TM, typename TV, int... I>
struct reference<declare::Field_<TM, TV, I...>> {
    typedef declare::Field_<TM, TV, I...> const& type;
};

template <typename TM, typename TV, int... I>
struct reference<const declare::Field_<TM, TV, I...>> {
    typedef declare::Field_<TM, TV, I...> const& type;
};

template <typename TM, typename TV, int... I>
struct value_type<declare::Field_<TM, TV, I...>> {
    typedef TV type;
};

template <typename TM, typename TV, int... I>
struct rank<declare::Field_<TM, TV, I...>> : public index_const<3> {};

template <typename TM, typename TV, int IFORM, int DOF, int... I>
struct iform<declare::Field_<TM, TV, IFORM, DOF, I...>> : public index_const<IFORM> {};

template <typename TM, typename TV, int IFORM, int DOF, int... I>
struct dof<declare::Field_<TM, TV, IFORM, DOF, I...>> : public index_const<DOF> {};

template <typename T>
struct field_value_type {
    typedef T type;
};

template <typename T>
using field_value_t = typename field_value_type<T>::type;

template <typename TM, typename TV, int IFORM, int DOF>
struct field_value_type<declare::Field_<TM, TV, IFORM, DOF>> {
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
                               declare::nTuple_<cell_tuple, 3>>
        type;
};

}  // namespace traits{

template <int... N>
struct PlaceHolder;
template <int N>
struct PlaceHolder<N> {
    int v = 0;

    PlaceHolder<N> operator-(int n) const { return PlaceHolder<N>{v - n}; }
};
static constexpr PlaceHolder<0> I{0};
static constexpr PlaceHolder<1> J{0};
static constexpr PlaceHolder<2> K{0};

template <typename TM, typename TV, int IFORM, int DOF>
class FieldView<TM, TV, IFORM, DOF> : public concept::Printable {
   private:
    typedef FieldView<TM, TV, IFORM, DOF> this_type;
    //    typedef TM::attribute<TV, IFORM, DOF> base_type;

   public:
    typedef TV value_type;

    typedef TM mesh_type;

    typedef typename mesh_traits<mesh_type>::id mesh_id;

   private:
    value_type* m_data_ = nullptr;

    std::shared_ptr<value_type> m_data_holder_ = nullptr;

    mesh_type const* m_mesh_;

    int m_sub_ = -1;
    mesh_id m_shift_;

   public:
    FieldView() : m_mesh_(nullptr), m_data_(nullptr){};

    explicit FieldView(this_type const& other)
        : m_data_(const_cast<value_type*>(other.data())),
          m_mesh_(other.mesh()),
          m_data_holder_(other.data_holder()),
          m_sub_(other.m_sub_),
          m_shift_(other.m_shift_) {}

    explicit FieldView(this_type const& other, int sub) : FieldView(other) { m_sub_ = sub; }

    template <int... N>
    explicit FieldView(this_type const& other, PlaceHolder<N...> const&) : FieldView(other) {}

    FieldView(mesh_type const* m, value_type* d = nullptr)
        : m_mesh_(m), m_data_(d), m_data_holder_(d, simpla::tags::do_nothing()) {}

    FieldView(mesh_type const* m, std::shared_ptr<value_type> const& d)
        : m_mesh_(m), m_data_(d.get()), m_data_holder_(d) {}

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
        return m_mesh_->size(IFORM, DOF);
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

    decltype(auto) operator[](mesh_id const& s) const { return at(s); }
    decltype(auto) operator[](mesh_id const& s) { return at(s); }
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

    //    template <typename... Args>
    //    void assign(mesh_id const& s, Args&&... args) {
    //        calculus_policy::assign(*m_mesh_, *this, s, std::forward<Args>(args)...);
    //    }
    template <typename... Args>
    void assign(Args&&... args) {
        apply(tags::_assign(), std::forward<Args>(args)...);
    }
    //**********************************************************************************************

    template <typename TOP, typename... Args>
    void apply_(Range<mesh_id> const& r, TOP const& op, Args&&... args) {
        int num_com = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);

        //        for (int i = 0; i < num_com; ++i) {
        //            for (int j = 0; j < DOF; ++j) {
        //                mesh::MeshEntityId tag = mesh::MeshEntityIdCoder::get_tag(IFORM, i, j);
        //
        //                r.foreach ([&](mesh_id s) {
        //                    s = mesh::MeshEntityIdCoder::tag(s, tag.v);
        //                    op(at(s), calculus_policy::get_value(*m_mesh_,
        //                    std::forward<Args>(args), s)...);
        //                });
        //            }
        //        }
    }

    template <typename... Args>
    void apply(Args&&... args) {
        apply_(m_mesh_->range(), std::forward<Args>(args)...);
    }

};  // class FieldView

namespace declare {

template <typename TM, typename TV, int IFORM, int DOF>
class Field_<TM, TV, IFORM, DOF> : public FieldView<TM, TV, IFORM, DOF> {
    typedef Field_<TM, TV, IFORM, DOF> this_type;
    typedef FieldView<TM, TV, IFORM, DOF> base_type;

   public:
    typedef TV value_type;

    template <typename... Args>
    explicit Field_(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    ~Field_() {}

    using typename base_type::mesh_id;
    using base_type::at;
    using base_type::get;
    using base_type::mesh;
    using base_type::deploy;
    using base_type::print;

    this_type operator[](int n) const { return Field_<TM, TV, IFORM, DOF>(*this, n); }

    template <int... N>
    this_type operator[](PlaceHolder<N...> const& s) const {
        return Field_<TM, TV, IFORM, DOF>(*this, s);
    }

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
