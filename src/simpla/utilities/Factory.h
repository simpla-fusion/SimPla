//
// Created by salmon on 17-4-12.
//

#ifndef SIMPLA_FACTORY_H
#define SIMPLA_FACTORY_H

#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include "Log.h"
#include "ObjectHead.h"
#include "SingletonHolder.h"
#include "type_traits.h"
namespace simpla {
template <typename TObj, typename... Args>
class Factory {
   public:
    Factory() = default;
    virtual ~Factory() = default;

    struct ObjectFactory {
        std::map<std::string, std::function<std::shared_ptr<TObj>(Args const &...)>> m_factory_;
    };
    static bool HasCreator(std::string const &k) {
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        return f.find(k) != f.end();
    }
    static std::string ShowDescription(std::string const &k = "") {
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        std::string res;
        if (!k.empty()) {
            auto it = f.find(k);
            if (it != f.end()) { res = it->first; }
        }
        if (res.empty()) {
            std::ostringstream os;
            os << std::endl << "Registered " << traits::type_name<TObj>::value() << " Creator:" << std::endl;
            for (auto const &item : f) { os << " " << item.first << std::endl; }
            res = os.str();
        }
        return res;
    };
    static int RegisterCreator(std::string const &k,
                               std::function<std::shared_ptr<TObj>(Args const &...)> const &fun) noexcept {
        return SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(k, fun).second ? 1 : 0;
    };
    template <typename U>
    static int RegisterCreator(std::string const &k_hint = "",
                               ENABLE_IF((std::is_constructible<U, Args...>::value))) noexcept {
        return RegisterCreator(!k_hint.empty() ? k_hint : U::RegisterName(),
                               [](Args const &... args) { return std::make_shared<U>(args...); });
    };
    template <typename U>
    static int RegisterCreator(std::string const &k_hint = "",
                               ENABLE_IF((!std::is_constructible<U, Args...>::value))) noexcept {
        return RegisterCreator(!k_hint.empty() ? k_hint : traits::type_name<U>::value(),
                               [](Args const &... args) { return U::New(args...); });
    };

   private:
    template <typename... U>
    static std::shared_ptr<TObj> _TryCreate(std::integral_constant<bool, true> _, U &&... args) {
        return std::make_shared<TObj>(std::forward<U>(args)...);
    }
    template <typename... U>
    static std::shared_ptr<TObj> _TryCreate(std::integral_constant<bool, false> _, U &&... args) {
        return nullptr;
    }

   public:
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::string const &k, U &&... args) {
        if (k.empty()) { return nullptr; }
        //        if (k.find("://") != std::string::npos) { return Create_(data::DataTable(k), args...); }
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        std::shared_ptr<TObj> res = nullptr;
        auto it = f.find(k);
        if (it != f.end()) {
            res = it->second(std::forward<U>(args)...);
        } else {
            RUNTIME_ERROR << "Can not find Creator \"" << k << "\"" << std::endl << ShowDescription() << std::endl;
        }
        return res;
    }
};
}  // namespace data{

#define SP_REGISTER_CREATOR(_BASE_NAME_, _CLASS_NAME_) \
    bool _CLASS_NAME_::_is_registered =                \
        simpla::Factory<_BASE_NAME_>::RegisterCreator<_CLASS_NAME_>(_CLASS_NAME_::RegisterName());

#define SP_CREATABLE_HEAD(_BASE_NAME_, _CLASS_NAME_, _REGISTER_NAME_)                                                  \
   public:                                                                                                             \
    static bool _is_registered;                                                                                        \
                                                                                                                       \
    static std::string RegisterName() { return __STRING(_REGISTER_NAME_); }                                            \
    std::string FancyTypeName() const override { return _BASE_NAME_::FancyTypeName() + "." + __STRING(_CLASS_NAME_); } \
                                                                                                                       \
   private:                                                                                                            \
    typedef _BASE_NAME_ base_type;                                                                                     \
    typedef _CLASS_NAME_ this_type;                                                                                    \
                                                                                                                       \
   protected:                                                                                                          \
    _CLASS_NAME_(_CLASS_NAME_ const &other);                                                                           \
    explicit _CLASS_NAME_(DataEntry::eNodeType etype = DN_TABLE);                                                      \
                                                                                                                       \
   public:                                                                                                             \
    ~_CLASS_NAME_() override;                                                                                          \
                                                                                                                       \
   public:                                                                                                             \
    template <typename... Args>                                                                                        \
    static std::shared_ptr<this_type> New(Args &&... args) {                                                           \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                                 \
    }                                                                                                                  \
    std::shared_ptr<DataEntry> Copy() const override { return std::shared_ptr<this_type>(new this_type(*this)); }      \
    std::shared_ptr<_CLASS_NAME_> Self() { return std::dynamic_pointer_cast<this_type>(this->shared_from_this()); };   \
    std::shared_ptr<const _CLASS_NAME_> Self() const {                                                                 \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type *>(this)->shared_from_this());                \
    };

template <typename T>
static bool RegisterCreator() {
    return T::template RegisterCreator<T>();
}  // namespace simpla{
#endif  // SIMPLA_FACTORY_H
