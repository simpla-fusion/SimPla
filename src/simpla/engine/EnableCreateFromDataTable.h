//
// Created by salmon on 17-4-12.
//

#ifndef SIMPLA_ENABLECREATEFROMDATATABLE_H
#define SIMPLA_ENABLECREATEFROMDATATABLE_H

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include "simpla/data/DataTable.h"
#include "simpla/engine/SPObject.h"
#include "simpla/utilities/SingletonHolder.h"
namespace simpla {
namespace engine {

template <typename TObj, typename... Args>
class EnableCreateFromDataTable : public engine::SPObject, public data::Serializable {

    SP_OBJECT_HEAD(EnableCreateFromDataTable<TObj>, engine::SPObject);

   public:
    explicit EnableCreateFromDataTable() = default;
    ~EnableCreateFromDataTable() override = default;
    EnableCreateFromDataTable(this_type const &other) = delete;
    EnableCreateFromDataTable(this_type &&other) = delete;
    this_type &operator=(this_type const &other) = delete;
    this_type &operator=(this_type &&other) = delete;

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = std::make_shared<data::DataTable>();
        res->SetValue("Type", GetFancyTypeName());
        return res;
    }

    void Deserialize(const std::shared_ptr<data::DataTable> &t) override{};

    struct ObjectFactory {
        std::map<std::string, std::function<TObj *(Args const &...)>> m_factory_;
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
            os << std::endl << "Registered " << TObj::GetFancyTypeName_s() << " Creator:" << std::endl;
            for (auto const &item : f) { os << " " << item.first << std::endl; }
            res = os.str();
        }
        return res;
    };
    static bool RegisterCreator(std::string const &k, std::function<TObj *(Args const &...)> const &fun) noexcept {
        return SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(k, fun).second;
    };
    template <typename U>
    static bool RegisterCreator(std::string const &k_hint = "") noexcept {
        return RegisterCreator(!k_hint.empty() ? k_hint : U::GetFancyTypeName_s(),
                               [](Args const &... args) { return new U(args...); });
    };
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::string const &k, U const &... args) {
        if (k.empty()) { return nullptr; }
        if (k.find("://") != std::string::npos) { return Create(std::make_shared<data::DataTable>(k), args...); }
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        std::shared_ptr<TObj> res = nullptr;
        auto it = f.find(k);

        if (it != f.end()) {
            res.reset(it->second(args...));
            LOGGER << TObj::GetFancyTypeName_s() << "::" << it->first << "  is created!" << std::endl;
        } else {
            std::ostringstream os;
            os << "Can not find Creator " << k << std::endl;
            os << std::endl << "Register " << TObj::GetFancyTypeName_s() << " Creator:" << std::endl;
            for (auto const &item : f) { os << item.first << std::endl; }
            WARNING << os.str();
        }
        return res;
    }

   private:
    template <typename... U>
    static std::shared_ptr<TObj> _CreateIfNotAbstract(std::integral_constant<bool, true> _, U &&... args) {
        return std::make_shared<TObj>(std::forward<U>(args)...);
    }
    template <typename... U>
    static std::shared_ptr<TObj> _CreateIfNotAbstract(std::integral_constant<bool, false> _, U &&... args) {
        return nullptr;
    }

   public:
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::shared_ptr<data::DataEntity> const &cfg, U &&... args) {
        std::shared_ptr<TObj> res = nullptr;
        std::string s_type;
        if (cfg == nullptr) {
        } else if (cfg->value_type_info() == typeid(std::string)) {
            s_type = data::DataCastTraits<std::string>::Get(cfg);
        } else if (cfg->isTable()) {
            auto t = std::dynamic_pointer_cast<data::DataTable>(cfg);
            s_type = t->GetValue<std::string>("Type", "");
        }

        if (!s_type.empty()) {
            res = Create(s_type, args...);
        } else {
            res = _CreateIfNotAbstract(std::integral_constant<bool, !std::is_abstract<TObj>::value>(),
                                       std::forward<U>(args)...);
        }

        if (res != nullptr && cfg != nullptr && cfg->isTable()) {
            res->Deserialize(std::dynamic_pointer_cast<data::DataTable>(cfg));
        }

        return res;
    }
};

#define REGISTER_CREATOR(_CLASS_NAME_, _REGISTER_NAME_) \
    bool _CLASS_NAME_::_is_registered = _CLASS_NAME_::RegisterCreator<_CLASS_NAME_>(__STRING(_REGISTER_NAME_));

}  // namespace data{
template <typename T>
static bool RegisterCreator(std::string const &name) {
    return T::template RegisterCreator<T>(name);
}
}  // namespace simpla{
#endif  // SIMPLA_ENABLECREATEFROMDATATABLE_H
