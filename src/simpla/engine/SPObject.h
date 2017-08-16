/**
 * @file object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_OBJECT_H
#define SIMPLA_OBJECT_H
#include "simpla/SIMPLA_config.h"

#include <memory>
#include <mutex>
#include <typeindex>
#include <typeinfo>

#include "simpla/data/DataEntity.h"
#include "simpla/data/DataTable.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/ObjectHead.h"
#include "simpla/utilities/Signal.h"

namespace simpla {
namespace data {
class DataTable;
}
namespace engine {
#define NULL_ID static_cast<id_type>(-1)

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
        state null{
        }
        state initialized {
        }
        state modified {
        }
        state ready{
        }

        null -->   initialized: Initialize
        initialized --> modified : Click
        modified --> ready    : Tag/Update
        ready  -->  modified  : Click
        ready --> initialized: TearDown
        modified --> null     : ResetTag/Finalize

    @enduml


 **/

class SPObject : public Factory<SPObject>, public std::enable_shared_from_this<SPObject> {
    SP_OBJECT_HEAD(SPObject, Factory<SPObject>)
    static constexpr char const *TagName() { return "SPObject"; }

   protected:
    SPObject();

   public:
    virtual ~SPObject();

    virtual void Serialize(std::shared_ptr<data::DataEntity> const &cfg) const;
    virtual void Deserialize(std::shared_ptr<const data::DataEntity> const &cfg);
    static std::shared_ptr<SPObject> New(std::shared_ptr<const data::DataEntity> const &v) {
        auto res = base_type::Create(std::string(TagName()) + "." + v->as<std::string>(""));
        res->Deserialize(v);
        return res;
    };
    const data::DataTable &db() const;
    data::DataTable &db();
    id_type GetGUID() const;
    void SetName(std::string const &);
    std::string const &GetName() const;

    void lock();
    void unlock();
    bool try_lock();
    void Tag();
    void Click();
    void ResetTag();
    size_type GetTagCount() const;
    size_type GetClickCount() const;
    bool isModified() const;
    bool isInitialized() const;

    virtual void DoInitialize();  //!< invoke once, before everything
    virtual void DoUpdate();      //!< repeat invoke, Update object after modified
    virtual void DoTearDown();    //!< repeat invoke,
    virtual void DoFinalize();    //!< invoke once, after everything

    design_pattern::Signal<void(SPObject *)> PreInitialize;
    design_pattern::Signal<void(SPObject *)> PostInitialize;
    design_pattern::Signal<void(SPObject *)> PreUpdate;
    design_pattern::Signal<void(SPObject *)> PostUpdate;
    design_pattern::Signal<void(SPObject *)> PreTearDown;
    design_pattern::Signal<void(SPObject *)> PostTearDown;
    design_pattern::Signal<void(SPObject *)> PreFinalize;
    design_pattern::Signal<void(SPObject *)> PostFinalize;

    void Initialize();
    void Finalize();
    void Update();
    void TearDown();

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

std::ostream &operator<<(std::ostream &os, SPObject const &obj);
std::istream &operator<<(std::istream &os, SPObject &obj);

std::ostream &operator<<(std::ostream &os, std::shared_ptr<const SPObject> const &obj);
std::istream &operator<<(std::istream &os, std::shared_ptr<SPObject> const &obj);

#define SP_OBJECT_DECLARE_MEMBERS(_CLASS_NAME_, _BASE_)                                                 \
    SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_)                                                                \
   protected:                                                                                           \
    _CLASS_NAME_();                                                                                     \
                                                                                                        \
   public:                                                                                              \
    ~_CLASS_NAME_() override;                                                                           \
    SP_DEFAULT_CONSTRUCT(_CLASS_NAME_);                                                                 \
                                                                                                        \
    void Serialize(std::shared_ptr<simpla::data::DataEntity> const &cfg) const override;                \
    void Deserialize(std::shared_ptr<simpla::data::DataEntity const> const &cfg) override;              \
                                                                                                        \
   private:                                                                                             \
    struct pimpl_s;                                                                                     \
    pimpl_s *m_pimpl_ = nullptr;                                                                        \
                                                                                                        \
    template <typename U, typename... Args>                                                             \
    static std::shared_ptr<U> _TryCreate(std::integral_constant<bool, false> _, Args &&... args) {      \
        return std::shared_ptr<U>(new U(std::forward<Args>(args)...));                                  \
    }                                                                                                   \
    template <typename U, typename... Args>                                                             \
    static std::shared_ptr<U> _TryCreate(std::integral_constant<bool, true> _, Args &&... args) {       \
        return std::dynamic_pointer_cast<_CLASS_NAME_>(base_type::Create(std::forward<Args>(args)...));    \
    }                                                                                                   \
                                                                                                        \
   public:                                                                                              \
    template <typename... Args>                                                                         \
    static std::shared_ptr<_CLASS_NAME_> New(Args &&... args) {                                         \
        return _TryCreate<_CLASS_NAME_>(std::is_abstract<_CLASS_NAME_>(), std::forward<Args>(args)...); \
    };

//
//    /**
//     * @brief Initial setup.
//       @startuml
//          title  TryInitialize()
//          (*) --> if "isInitialized()?" then
//                      --> [true] (*)
//                  else
//                      --> [false] "state = INITIALIZED"
//                      --> (*)
//
//                 endif
//      @enduml
//     */
//    bool TryInitialize();
//    /**
//     *  @brief Initial setup.
//     */
//    virtual void InitializeConditionPatch();
//
//    /**
//     * @brief Initial setup. This function should be invoked _ONLY ONCE_  after InitializeConditionPatch()
//     * @startuml
//     *    title  TryPreProcess()
//     *    (*) --> if "isPrepared()?" then
//     *                --> [true] (*)
//     *            else
//     *                --> [false] InitializeConditionPatch()
//     *                --> "state = PREPARED"
//     *                --> (*)
//     *           endif
//     * @enduml
//    */
//    bool TryPreProcess();
//    virtual void PreProcess();  //< This function should be called before operation
//
//    /**
//     * @startuml
//     *    title  lock()
//     *    (*) -down-> START
//     *            if "isLocked()?" then
//     *                --> [true] wait()
//     *                -up-> START
//     *            else
//     *                 -left-> [false] " state=LOCKED"
//     *                 --> (*)
//     *           endif
//     * @enduml
//     */
//    virtual void Lock();
//
//    /**
//    * @startuml
//    *    title  TryLock()
//    *    (*) --> check
//     *           if "isLocked()?" then
//    *                -left->   [true] check
//    *            else
//     *                --> PreProcess()
//    *                 --> [false] " state=LOCKED"
//    *                 --> (*)
//    *           endif
//    * @enduml
//    */
//    virtual bool TryLock();
//
//    /**
//     * @startuml
//     *    title  Unlock()
//     *    (*) --> if "isLocked()?" then
//     *                --> [true] "--state"
//     *                --> (*)
//     *            else
//     *                --> [false]   (*)
//     *           endif
//     * @enduml
//     */
//    virtual void Unlock();
//
//    /**
//     * @brief   This function should be called after operation
//     * @startuml
//     *    title  PostProcess()
//     *    (*) --> if "isPrepared()?" then
//     *                --> [true]  Unlock()
//     *                --> "--state "
//     *                --> (*)
//     *            else
//     *                --> [false]   (*)
//     *           endif
//     * @enduml
//     */
//    bool TryPostProcess();
//    virtual void PostProcess();
//
//    /**
//     * @brief Finalize object. This function should be invoked _ONLY ONCE_ before DoFinalize()
//     * @startuml
//     * title  DoFinalize()
//     * (*) --> if "isInitialized()?" then
//     *     -->[true] PostProcess()
//     *     --> "--state "
//     *     --> (*)
//     * else
//     *     -->[false] (*)
//     * endif
//     * @enduml
//     *
//     *
//     */
//
//    virtual void DoFinalize();
//    bool TryFinalize();
/** @} */
}  // namespace engine{
}  // namespace simpla { namespace base

#endif  // SIMPLA_OBJECT_H
