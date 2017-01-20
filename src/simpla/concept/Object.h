/**
 * @file object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_OBJECT_H
#define SIMPLA_OBJECT_H

#include <memory>
#include <mutex>
#include <typeinfo>

#include <simpla/SIMPLA_config.h>

#include <typeindex>

namespace simpla {

#define SP_OBJECT_BASE(_BASE_CLASS_NAME_)                                                            \
   private:                                                                                          \
    typedef _BASE_CLASS_NAME_ this_type;                                                             \
                                                                                                     \
   public:                                                                                           \
    virtual bool isA(const std::type_info &info) const { return typeid(_BASE_CLASS_NAME_) == info; } \
    template <typename _UOTHER_>                                                                     \
    bool isA() const {                                                                               \
        return isA(typeid(_UOTHER_));                                                                \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ *as() {                                                                                       \
        return (isA(typeid(U_))) ? static_cast<U_ *>(this) : nullptr;                                \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ const *as() const {                                                                           \
        return (isA(typeid(U_))) ? static_cast<U_ const *>(this) : nullptr;                          \
    }                                                                                                \
    virtual std::type_index TypeIndex() const { return std::type_index(typeid(_BASE_CLASS_NAME_)); } \
    virtual std::string getClassName() const { return __STRING(_BASE_CLASS_NAME_); }

#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                                         \
   public:                                                                                      \
    virtual bool isA(std::type_info const &info) const {                                        \
        return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::isA(info);                    \
    }                                                                                           \
    virtual std::type_index TypeIndex() const { return std::type_index(typeid(_CLASS_NAME_)); } \
    virtual std::string getClassName() const { return __STRING(_CLASS_NAME_); }                 \
                                                                                                \
   private:                                                                                     \
    typedef _BASE_CLASS_NAME_ base_type;                                                        \
    typedef _CLASS_NAME_ this_type;                                                             \
                                                                                                \
   public:

/**
 *  @addtogroup concept
 *
 *  @brief Object
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
 *
 * ## Summary
 * Requirements for a type whose instances share ownership between multiple objects;
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref LifeControllable must define:
 *   Pseudo-Signature                   | Semantics
 *	 -----------------------------------|----------
 * 	 \code   R()               \endcode | Constructor;
 * 	 \code  virtual ~R()       \endcode | Destructor
 *   \code bool isNull()       \endcode |  return m_state_ == NULL_STATE
 *   \code bool isBlank()      \endcode |  return m_state_ == BLANK
 *   \code bool isValid()      \endcode |  return m_state_ == VALID
 *   \code bool isReady()      \endcode |  return m_state_ == READY
 *   \code bool isLocked()     \endcode |  return m_state_ == LOCKED
 * 	 \code void Deploy()       \endcode |  allocate memory
 *   \code void Initialize()   \endcode |  Initial setup. This function should be invoked ONLY ONCE after Deploy()
 *   \code void PreProcess()   \endcode |  This function should be called before operation
 *   \code void Lock()         \endcode |  lock object for concurrent operation
 *   \code bool TryLock()      \endcode |  return true if object is locked
 *   \code void Unlock()       \endcode |  unlock object after concurrent operation
 *   \code void PostProcess()  \endcode |  This function should be called after operation
 *   \code void Finalize()     \endcode |  Finalize object. This function should be invoked ONLY ONCE  before Destroy()
 *   \code void Destroy()      \endcode |  release memory
 *
 *
  @startuml
        [*] -->Null         : Construct
        Null --> Blank      : Deploy
        Blank --> Valid     : Initialize
        Valid --> Ready     : PreProcess
        Ready --> Locked    : lock
        Locked  -->  Ready  : unlock
        Ready --> Valid     : PostProcess
        Valid --> Blank     : Finalize
        Blank --> Null      : Destroy
        Null --> [*]        : Destruct

   @enduml
 **/

class Object {
   public:
    SP_OBJECT_BASE(Object)
   public:
    enum { NULL_STATE = 0, BLANK, VALID, READY, LOCKED };
    Object();

    Object(Object &&other);

    Object(Object const &) = delete;

    Object &operator=(Object const &other) = delete;

    virtual ~Object();

    void id(id_type t_id);

    id_type id() const;

    bool operator==(Object const &other);

    void lock();
    void unlock();
    bool try_lock();
    void touch();
    size_type click() const;

    unsigned int state() const { return m_state_; }
    bool isNull() const { return m_state_ == NULL_STATE; }
    bool isBlank() const { return m_state_ == BLANK; }
    bool isValid() const { return m_state_ >= VALID; }
    bool isReady() const { return m_state_ == READY; }
    bool isLocked() const { return m_state_ == LOCKED; }
    bool isDeployed() const { return m_state_ >= BLANK; }

    virtual void Deploy();      //< Initial setup. This function should be invoked ONLY ONCE
    virtual void Initialize();  //< Initial setup. This function should be invoked _ONLY ONCE_  after Deploy()
    virtual void PreProcess();  //< This function should be called before operation
    virtual void Lock();
    virtual bool TryLock();
    virtual void Unlock();
    virtual void PostProcess();  //< This function should be called after operation
    virtual void Finalize();     //< Finalize object. This function should be invoked _ONLY ONCE_ before Destroy()
    virtual void Destroy();

   private:
    unsigned int m_state_ = NULL_STATE;

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace simpla { namespace base

#endif  // SIMPLA_OBJECT_H
