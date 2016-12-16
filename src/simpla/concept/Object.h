/**
 * @file object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_OBJECT_H
#define SIMPLA_OBJECT_H

#include <typeinfo>
#include <mutex>
#include <memory>

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/LifeClick.h>
#include <typeindex>

namespace simpla
{

#define SP_OBJECT_BASE(_BASE_CLASS_NAME_)                                                                           \
virtual bool is_a(const std::type_info &info) const { return typeid(_BASE_CLASS_NAME_) == info; }                   \
template<typename _UOTHER_> bool is_a()const {return is_a(typeid(_UOTHER_));}                                        \
template<typename U_> U_ * as() { return (is_a(typeid(U_))) ? static_cast<U_ *>(this) : nullptr; }                   \
template<typename U_> U_ const * as() const { return (is_a(typeid(U_))) ? static_cast<U_ const *>(this) : nullptr; } \
virtual std::type_index typeindex() const   { return std::type_index(typeid(_BASE_CLASS_NAME_)); }                  \
virtual std::string get_class_name() const { return __STRING(_BASE_CLASS_NAME_); }                                  \
private:                                                                                                            \
typedef _BASE_CLASS_NAME_ this_type;                                                                                \
public:

#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                       \
virtual bool is_a(std::type_info const &info)const { return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::is_a(info); }   \
virtual std::type_index typeindex() const   { return std::type_index(typeid(_CLASS_NAME_)); }  \
virtual std::string get_class_name() const { return __STRING(_CLASS_NAME_); } \
private:                                                                      \
    typedef _BASE_CLASS_NAME_ base_type;                                      \
    typedef _CLASS_NAME_ this_type;                                           \
public:


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
 *   `sync()`					| allocate memory
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
    SP_OBJECT_BASE(Object)

    Object();

    Object(Object &&other);

    Object(Object const &) = delete;

    Object &operator=(Object const &other)= delete;

    virtual  ~Object();

    void id(id_type t_id);

    id_type id() const;

    bool operator==(Object const &other);

    /**
     *  @name concept lockable
     *  @{
     */

    void lock();

    void unlock();

    bool try_lock();

    /** @} */

    /**
     *  @name concept touch count
     *  @{
     */
    void touch();

    size_type click() const;
    /** @} */
private:

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};



//virtual std::shared_ptr<GeoObject> clone_object()const { return std::dynamic_pointer_cast<GeoObject>(this->clone()); }



}//namespace simpla { namespace base


#endif //SIMPLA_OBJECT_H
