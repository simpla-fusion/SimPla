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
/**
 * @addtogroup concept
 * @{
 */

/**
 * @brief  define the common part of the base class
 */
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

/**
 * @brief define the common part of the derived class
 */
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
 *
 *  @brief every thing is an Object
 *
 *  @details
 *
 *  ## Summary
 *    Life cycle control of an object
 *
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Object must define:
 *   Pseudo-Signature                   | Semantics
 *	 -----------------------------------|----------
 * 	 \code R()                 \endcode | Constructor;
 * 	 \code virtual ~R()        \endcode | Destructor
 * 	 \code id_type id()        \endcode |  return id
 *   \code bool isNull()       \endcode |  return m_state_ == NULL_STATE
 *   \code bool isValid()      \endcode |  return m_state_ == VALID
 *   \code bool isReady()      \endcode |  return m_state_ == READY
 *   \code bool isLocked()     \endcode |  return m_state_ == LOCKED
 *   \code void Initialize()   \endcode |  Initial setup. This function should be invoked ONLY ONCE after Deploy()
 *   \code void PreProcess()   \endcode |  This function should be called before operation
 *   \code void Lock()         \endcode |  lock object for concurrent operation
 *   \code bool TryLock()      \endcode |  return true if object is locked
 *   \code void Unlock()       \endcode |  unlock object after concurrent operation
 *   \code void PostProcess()  \endcode |  This function should be called after operation
 *   \code void Finalize()     \endcode |  Finalize object. This function should be invoked ONLY ONCE  before Destroy()
 *
 *
  @startuml
         state initialized         {
           state prepared         {
                state locked{
                }
            }
            }
        null -->   initialized: Initialize
        initialized --> prepared     : PreProcess
        prepared --> locked    : lock
        locked  -->  prepared  : unlock
        prepared --> initialized     : PostProcess
        initialized --> null     : Finalize
    @enduml

 **/

class Object {
   public:
    SP_OBJECT_BASE(Object)
   public:
    enum { NULL_STATE = 0, INITIALIZED, PREPARED, LOCKED };
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
    bool isInitialized() const { return m_state_ >= INITIALIZED; }
    bool isPrepared() const { return m_state_ >= PREPARED; }
    bool isLocked() const { return m_state_ == LOCKED; }
    unsigned int NextState();
    unsigned int PrevState();

    /**
     * @brief Initial setup.
       @startuml
          title  TryInitialize()
          (*) --> if "isInitialized()?" then
                      --> [true] (*)
                  else
                      --> [false] "state = INITIALIZED"
                      --> (*)

                 endif
      @enduml
     */
    bool TryInitialize();
    /**
     *  @brief Initial setup.
     */
    virtual void Initialize();

    /**
     * @brief Initial setup. This function should be invoked _ONLY ONCE_  after Initialize()
     * @startuml
     *    title  TryPreProcess()
     *    (*) --> if "isPrepared()?" then
     *                --> [true] (*)
     *            else
     *                --> [false] Initialize()
     *                --> "state = PREPARED"
     *                --> (*)
     *           endif
     * @enduml
    */
    bool TryPreProcess();
    virtual void PreProcess();  //< This function should be called before operation

    /**
     * @startuml
     *    title  lock()
     *    (*) -down-> START
     *            if "isLocked()?" then
     *                --> [true] wait()
     *                -up-> START
     *            else
     *                 -left-> [false] " state=LOCKED"
     *                 --> (*)
     *           endif
     * @enduml
     */
    virtual void Lock();

    /**
    * @startuml
    *    title  TryLock()
    *    (*) --> check
     *           if "isLocked()?" then
    *                -left->   [true] check
    *            else
     *                --> PreProcess()
    *                 --> [false] " state=LOCKED"
    *                 --> (*)
    *           endif
    * @enduml
    */
    virtual bool TryLock();

    /**
     * @startuml
     *    title  Unlock()
     *    (*) --> if "isLocked()?" then
     *                --> [true] "--state"
     *                --> (*)
     *            else
     *                --> [false]   (*)
     *           endif
     * @enduml
     */
    virtual void Unlock();

    /**
     * @brief   This function should be called after operation
     * @startuml
     *    title  PostProcess()
     *    (*) --> if "isPrepared()?" then
     *                --> [true]  Unlock()
     *                --> "--state "
     *                --> (*)
     *            else
     *                --> [false]   (*)
     *           endif
     * @enduml
     */
    bool TryPostProcess();
    virtual void PostProcess();

    /**
     * @brief Finalize object. This function should be invoked _ONLY ONCE_ before TearDown()
     * @startuml
     * title  Finalize()
     * (*) --> if "isInitialized()?" then
     *     -->[true] PostProcess()
     *     --> "--state "
     *     --> (*)
     * else
     *     -->[false] (*)
     * endif
     * @enduml
     *
     *
     */

    virtual void Finalize();
    bool TryFinalize();

   private:
    unsigned int m_state_ = NULL_STATE;

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
/** @} */
}  // namespace simpla { namespace base

#endif  // SIMPLA_OBJECT_H
