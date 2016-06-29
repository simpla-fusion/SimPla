/**
 * @file object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_OBJECT_H
#define SIMPLA_OBJECT_H

#include <typeinfo>
#include <mutex>
#include <stddef.h> //for size_t
#include <memory>

#include <boost/uuid/uuid.hpp>


#include  "LifeClick.h"
#include "../gtl/design_pattern/Visitor.h"

namespace simpla { namespace base
{


/** @ingroup task_flow
 *  @addtogroup sp_object SIMPla object
 *  @{
 *  ## Summary
 *   - Particle distribution function is a @ref physical_object;
 *   - Electric field is a @ref physical_object
 *   - Magnetic field is a @ref physical_object;
 *   - Plasma density field is a @ref physical_object;
 *   - @ref physical_object is a geometry defined on a domain in configuration space;
 *   - @ref physical_object has properties;
 *   - @ref physical_object can be saved or loaded as dataset;
 *   - @ref physical_object may be decomposed and sync between mpi processes;
 *   - The element value of PhysicalObject may be accessed through a index of discrete grid point in the domain
 *
 *
 *  ## Member types
 *   Member type	 			| Semantics
 *   ---------------------------|--------------
 *   domain_type				| Domain
 *   iterator_type				| iterator of element value
 *   range_type					| entity_id_range of element value
 *
 *
 *
 *  ## Member functions
 *
 *  ### Constructor
 *
 *   Pseudo-Signature 	 			| Semantics
 *   -------------------------------|--------------
 *   `PhysicalObject()`						| Default constructor
 *   `~PhysicalObject() `					| destructor.
 *   `PhysicalObject( const PhysicalObject& ) `	| copy constructor.
 *   `PhysicalObject( PhysicalObject && ) `			| move constructor.
 *   `PhysicalObject( Domain & D ) `			| Construct a PhysicalObject on domain \f$D\f$.
 *
 *  ### Swap
 *    `swap(PhysicalObject & )`					| swap
 *
 *  ###  Fuctions
 *   Pseudo-Signature 	 			| Semantics
 *   -------------------------------|--------------
 *   `bool is_valid() `  			| _true_ if PhysicalObject is valid for accessing
 *   `update()`					| allocate memory
 *   `DataModel()`					| return the m_data set of PhysicalObject
 *   `clear()`						| set value to zero, allocate memory if empty() is _true_
 *   `T properties(std::string name)const` | get properties[name]
 *   `properties(std::string name,T const & v) ` | set properties[name]
 *   `std::ostream& print(std::ostream & os) const` | print description to `os`
 *
 *  ### Element access
 *   Pseudo-Signature 				| Semantics
 *   -------------------------------|--------------
 *   `value_type & at(index_type s)`   			| access element on the grid points _s_ with bounds checking
 *   `value_type & operator[](index_type s) `  | access element on the grid points _s_as
 *
 *  @}
 **/


class Object
{
public:
    Object();

    Object(Object &&other);

    Object(Object const &);

    Object &operator=(Object const &other);

    virtual  ~Object();

    void swap(Object &other);

    virtual void deploy() { }

    virtual bool is_a(std::type_info const &info) const;

    template<typename T> inline bool is_a() const { return is_a(typeid(T)); };

    virtual std::string get_class_name() const;

    virtual std::ostream &print(std::ostream &os, int indent) const;

    std::string name() const { return m_name_ == "" ? type_cast<std::string>(short_id()) : m_name_; };

    Object &name(std::string const &s)
    {
        m_name_ = s;
        return *this;
    };

    boost::uuids::uuid uuid() const { return m_uuid_; }

    boost::uuids::uuid id() const { return m_uuid_; }

    size_t short_id() const { return static_cast<size_t>(boost::uuids::hash_value(m_uuid_)); }

    bool operator==(Object const &other) { return m_uuid_ == other.m_uuid_; }
    /**
     *  @name concept lockable
     *  @{
     */
public:
    inline void lock() { m_mutex_.lock(); }

    inline void unlock() { m_mutex_.unlock(); }

    inline bool try_lock() { return m_mutex_.try_lock(); }

private:
    std::string m_name_{""};
    std::mutex m_mutex_;
    boost::uuids::uuid m_uuid_;

    /** @} */
public:
    /**
     *  @name concept touchable
     *  @{
     */
    inline void touch() { GLOBAL_CLICK_TOUCH(&m_click_); }

    inline size_t click() const { return m_click_; }
    /** @} */
private:

    size_t m_click_ = 0;
};

#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                       \
virtual bool is_a(std::type_info const &info)const                            \
  { return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::is_a(info); }   \
template<typename _UOTHER_> bool is_a()const {return is_a(typeid(_UOTHER_));} \
virtual std::string get_class_name() const { return __STRING(_CLASS_NAME_); } \
static std::string class_name()  { return __STRING(_CLASS_NAME_); }           \
private:                                                                      \
    typedef _BASE_CLASS_NAME_ base_type;                                      \
public:

//virtual std::shared_ptr<GeoObject> clone_object()const { return std::dynamic_pointer_cast<GeoObject>(this->clone()); }


}}//namespace simpla { namespace base


#endif //SIMPLA_OBJECT_H
