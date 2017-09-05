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

#include "simpla/data/DataNode.h"
#include "simpla/utilities/Factory.h"

namespace simpla {
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
    typedef Factory<SPObject> base_type;
    typedef SPObject this_type;

   public:
    static std::string GetFancyTypeName_s() { return "SPObject"; }
    virtual std::string GetFancyTypeName() const { return GetFancyTypeName_s(); }
    template <typename U>
    static int RegisterCreator() noexcept {
        return Factory<SPObject>::RegisterCreator<U>(U::GetFancyTypeName_s());
    };

   protected:
    SPObject();

   public:
    ~SPObject() override;

    virtual std::shared_ptr<data::DataNode> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataNode> const &);
    static std::shared_ptr<SPObject> Create(std::string const &v);
    static std::shared_ptr<SPObject> Create(std::shared_ptr<data::DataNode> const &tdb);
    static std::shared_ptr<SPObject> CreateAndSync(std::shared_ptr<data::DataNode> const &v);

    template <typename TOBJ>
    static std::shared_ptr<TOBJ> CreateAndSyncT(std::shared_ptr<data::DataNode> const &v) {
        return std::dynamic_pointer_cast<TOBJ>(CreateAndSync(v));
    };

    std::shared_ptr<data::DataNode> db() const;
    std::shared_ptr<data::DataNode> db();
    id_type GetGUID() const;
    void SetName(std::string const &);
    std::string const &GetName() const;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

std::ostream &operator<<(std::ostream &os, SPObject const &obj);
std::istream &operator>>(std::istream &is, SPObject &obj);

#define SP_OBJECT_PROPERTY(_TYPE_, _NAME_)                                  \
    void Set##_NAME_(_TYPE_ const & _v_) { db()->SetValue(__STRING(_NAME_), _v_); } \
    _TYPE_ Get##_NAME_() const { return db()->template GetValue<_TYPE_>(__STRING(_NAME_)); }

#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_)                                                      \
   public:                                                                                             \
    static std::string GetFancyTypeName_s() {                                                          \
        return _BASE_NAME_::GetFancyTypeName_s() + "." + __STRING(_CLASS_NAME_);                       \
    }                                                                                                  \
    virtual std::string GetFancyTypeName() const override { return GetFancyTypeName_s(); }             \
    static bool _is_registered;                                                                        \
                                                                                                       \
   private:                                                                                            \
    typedef _BASE_NAME_ base_type;                                                                     \
    typedef _CLASS_NAME_ this_type;                                                                    \
    struct pimpl_s;                                                                                    \
    pimpl_s *m_pimpl_ = nullptr;                                                                       \
                                                                                                       \
   public:                                                                                             \
   protected:                                                                                          \
    _CLASS_NAME_();                                                                                    \
                                                                                                       \
   public:                                                                                             \
    ~_CLASS_NAME_() override;                                                                          \
                                                                                                       \
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;                                \
    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;                     \
                                                                                                       \
   private:                                                                                            \
    template <typename U, typename... Args>                                                            \
    static std::shared_ptr<U> New_(std::false_type, Args &&... args) {                                 \
        return std::shared_ptr<U>(new U(std::forward<Args>(args)...));                                 \
    };                                                                                                 \
    template <typename U, typename... Args>                                                            \
    static std::shared_ptr<U> New_(std::true_type, Args &&... args) {                                  \
        return nullptr;                                                                                \
    };                                                                                                 \
                                                                                                       \
   public:                                                                                             \
    template <typename... Args>                                                                        \
    static std::shared_ptr<this_type> New(Args &&... args) {                                           \
        return New_<this_type>(std::is_abstract<this_type>(), std::forward<Args>(args)...);            \
    };                                                                                                 \
    static std::shared_ptr<this_type> New() { return New_<this_type>(std::is_abstract<this_type>()); } \
    static std::shared_ptr<this_type> New(std::shared_ptr<simpla::data::DataNode> const &cfg) {        \
        auto res = New();                                                                              \
        res->Deserialize(cfg);                                                                         \
        return res;                                                                                    \
    }

#define SP_OBJECT_REGISTER(_CLASS_NAME_) \
    bool _CLASS_NAME_::_is_registered = simpla::SPObject::RegisterCreator<_CLASS_NAME_>();

//    static std::shared_ptr<_CLASS_NAME_> New(std::shared_ptr<data::DataNode> v) {                                \
//        auto s_type = v->as<std::string>("");                                                                          \
//        if (s_type.empty() && std::dynamic_pointer_cast<const data::DataTable>(v) != nullptr) {                        \
//            s_type = std::dynamic_pointer_cast<const data::DataTable>(v)->GetEntity<std::string>("Type", "");           \
//        }                                                                                                              \
//        auto res = base_type::Create(std::string(TagName()) + (s_type.empty() ? "" : ".") + s_type);                   \
//        res->Deserialize(v);                                                                                           \
//        return std::dynamic_pointer_cast<_CLASS_NAME_>(res);                                                           \
//    };

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
}  // namespace simpla

#endif  // SIMPLA_OBJECT_H
