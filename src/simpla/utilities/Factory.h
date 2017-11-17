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

#define SP_REGISTER_HEAD(_REGISTER_NAME_) \
   private:                               \
    static bool _is_registered;           \
                                          \
   public:                                \
    static std::string RegisterName() { return __STRING(_REGISTER_NAME_); }

#define SP_REGISTER_CREATOR(_BASE_NAME_, _CLASS_NAME_) \
    bool _CLASS_NAME_::_is_registered =                \
        simpla::Factory<_BASE_NAME_>::RegisterCreator<_CLASS_NAME_>(_CLASS_NAME_::RegisterName());

template <typename T>
static bool RegisterCreator() {
    return T::template RegisterCreator<T>();
}  // namespace simpla{
#endif  // SIMPLA_FACTORY_H
