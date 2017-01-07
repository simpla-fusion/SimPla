/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>

#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/mesh/Attribute.h>
#include <cassert>
#include <type_traits>
#include "Algebra.h"
#include "nTuple.h"

namespace simpla {
namespace algebra {
namespace declare {
template <typename, typename, size_type...>
struct Field_;
}

namespace traits {

//***********************************************************************************************************************

template <typename>
struct mesh_type {
    typedef void type;
};

template <typename TV, typename TM, size_type... I>
struct mesh_type<declare::Field_<TV, TM, I...>> {
    typedef TM type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TV, TM, IFORM, DOF>> : public std::integral_constant<bool, true> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef declare::Field_<TV, TM, IFORM, DOF> const& type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<const declare::Field_<TV, TM, IFORM, DOF>> {
    typedef declare::Field_<TV, TM, IFORM, DOF> const& type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct value_type<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef TV type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct rank<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<3> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct iform<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<IFORM> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct dof<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<DOF> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct field_value_type<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
                               declare::nTuple_<cell_tuple, 3>>
        type;
};

template <typename V>
struct field_traits {
    typedef declare::Field_<V, typename mesh_type<V>::type, iform<V>::value, dof<V>::value> type;
    typedef calculus::calculator<type> calculator;
};
//
//template <typename T>
//struct primary_type<T, typename std::enable_if<is_field<T>::value>::type> {
//    typedef typename declare::Field_<value_type_t<T>, typename mesh_type<T>::type, iform<T>::value,
//                                     dof<T>::value>
//        type;
//};
template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct data_block_type {
    typedef declare::Array_<
        TV, rank<TM>::value + ((IFORM == VERTEX || IFORM == VOLUME ? 0 : 1) * DOF > 1 ? 1 : 0)>
        type;
};
template <typename TV, typename TM, size_type IFORM, size_type DOF>
using data_block_t = typename data_block_type<TV, TM, IFORM, DOF>::type;

}  // namespace traits{

namespace declare {
template <typename TV, typename TM, size_type IFORM, size_type DOF>
class Field_<TV, TM, IFORM, DOF> : public mesh::Attribute {
   private:
    typedef Field_<TV, TM, IFORM, DOF> this_type;

   public:
    typedef traits::field_value_t<this_type> field_value;
    typedef TV value_type;
    typedef TM mesh_type;

   private:
    typedef typename mesh_type::id_type mesh_id_type;

    typedef calculus::calculator<this_type> calculus_policy;

    friend calculus_policy;

    typedef typename calculus_policy::data_block_type data_type;

    data_type* m_data_;

    mesh_type const* m_mesh_;

   public:
    std::shared_ptr<data_type> m_data_holder_;

    Field_() : m_data_holder_(nullptr), m_mesh_(nullptr), m_data_(nullptr){};

    template <typename... Args>
    explicit Field_(Args&&... args)
        : mesh::Attribute(nullptr,
                          std::make_shared<mesh::AttributeDescTemp<value_type, IFORM, DOF>>(
                              std::forward<Args>(args)...)),
          m_mesh_(nullptr),
          m_data_holder_(nullptr),
          m_data_(nullptr) {}

    explicit Field_(mesh_type const* m, data_type* d)
        : m_mesh_(m), m_data_holder_(d, simpla::tags::do_nothing()), m_data_(nullptr){};

    explicit Field_(mesh_type const* m, std::shared_ptr<data_type> const& d = nullptr)
        : m_mesh_(m), m_data_holder_(d), m_data_(d.get()){};

    template <typename... Others>
    explicit Field_(mesh_type const& m, Others&&... others)
        : Field_(&m, std::forward<Others>(others)...) {}

    template <typename... Others>
    explicit Field_(std::shared_ptr<mesh_type> const& m, Others&&... others)
        : Field_(m.get(), std::forward<Others>(others)...) {}

    virtual ~Field_() {}

    Field_(this_type const& other) = delete;

    Field_(this_type&& other) = delete;

    virtual std::ostream& print(std::ostream& os, int indent = 0) const {
        return m_data_->print(os, indent);
    }

    virtual void load(data::DataTable const& d) { m_data_->load(d); };

    virtual void save(data::DataTable* d) const { m_data_->save(d); };

    virtual std::shared_ptr<mesh::DataBlock> create_data_block(mesh::MeshBlock const* m,
                                                               void* p = nullptr) const {
        return mesh::DataBlockAdapter<this_type>::create(m, static_cast<value_type*>(p));
    };
    inline this_type& operator=(this_type const& rhs) { return assign(rhs); }

    template <typename TR>
    inline this_type& operator=(TR const& rhs) {
        return assign(rhs);
    }

    virtual void pre_process() {
        deploy();
        assert(m_data_holder_ != nullptr);
        assert(m_mesh_ != nullptr);
    }

    virtual void post_process() {
        m_mesh_ = nullptr;
        m_data_ = nullptr;
        m_data_holder_.reset();
    }

    virtual void accept(mesh_type const* m, std::shared_ptr<data_type> const& d = nullptr) {
        post_process();
        m_data_holder_ = d;
        m_mesh_ = m;
        pre_process();
    }

    virtual void accept(mesh_type const* m, data_type* d) {
        post_process();
        m_data_holder_ = std::shared_ptr<data_type>(d, simpla::tags::do_nothing());
        m_mesh_ = m;
        pre_process();
    }

    virtual data_type* data() { return m_data_; }

    virtual data_type const* data() const { return m_data_; }

    virtual void deploy() { calculus_policy::deploy(*this); }

    virtual void reset() { calculus_policy::reset(*this); }

    virtual void clear() { calculus_policy::clear(*this); }

    /** @name as_function  @{*/
    template <typename... Args>
    inline auto gather(Args&&... args) const {
        return apply(tags::_gather(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto scatter(field_value const& v, Args&&... args) {
        return (apply(tags::_scatter(), v, std::forward<Args>(args)...));
    }

    //    auto
    //    operator()(mesh_type::point_type const &x) const AUTO_RETURN((gather(x)))

    /**@}*/

    /** @name as_array   @{*/
    value_type& at(mesh_id_type const& s) {
        return calculus_policy::get_value(*m_mesh_, *m_data_, s);
    }

    value_type const& at(mesh_id_type const& s) const {
        return calculus_policy::get_value(*m_mesh_, *m_data_, s);
    }

    inline value_type& operator[](mesh_id_type const& s) { return at(s); }

    inline value_type const& operator[](mesh_id_type const& s) const { return at(s); }

    template <typename... TID>
    value_type& at(TID&&... s) {
        return m_data_->at(std::forward<TID>(s)...);
    }

    template <typename... TID>
    value_type const& at(TID&&... s) const {
        return m_data_->at(std::forward<TID>(s)...);
    }

    //    template <typename... Args>
    //    value_type& operator()(Args&&... args) {
    //        return ((at(std::forward<Args>(args)...)));
    //    }
    //
    //    template <typename... Args>
    //    value_type const& operator()(Args&&... args) const {
    //        return ((at(std::forward<Args>(args)...)));
    //    }

    /**@}*/

    template <typename... Args>
    this_type& assign(Args&&... args) {
        return apply(tags::_assign(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    this_type& apply(Args&&... args) {
        pre_process();
        calculus_policy::apply(*this, *m_mesh_,   std::forward<Args>(args)...);
        return *this;
    }


};  // class Field_
}  // namespace declare
}
}  // namespace simpla::algebra

namespace simpla {
template <typename TV, typename TM, size_type IFORM = VERTEX, size_type DOF = 1>
using Field = algebra::declare::Field_<TV, TM, IFORM, DOF>;
}

#endif  // SIMPLA_FIELD_H
