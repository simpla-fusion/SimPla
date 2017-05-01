//
// Created by salmon on 17-4-12.
//

#ifndef SIMPLA_ENABLECREATEFROMDATATABLE_H
#define SIMPLA_ENABLECREATEFROMDATATABLE_H

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "DataTable.h"
#include "simpla/concept/CheckConcept.h"
#include "simpla/utilities/SingletonHolder.h"
namespace simpla {
namespace data {
class DataTable;
template <typename TObj, typename... Args>
class EnableCreateFromDataTable {
    typedef EnableCreateFromDataTable<TObj> this_type;

   public:
    EnableCreateFromDataTable() = default;
    virtual ~EnableCreateFromDataTable() = default;
    SP_DEFAULT_CONSTRUCT(EnableCreateFromDataTable);

    virtual std::string GetRegisterName() const { return TObj::RegisterName(); }

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
        auto it = f.find(k);
        if (it == f.end()) { it = f.begin(); }
        if (it != f.end()) {
            res = it->first;
        } else {
            std::ostringstream os;
            os << std::endl << "Register " << TObj::RegisterName() << " Creator:" << std::endl;
            for (auto const &item : f) { os << std::setw(15) << item.first << std::endl; }
            res = os.str();
        }
        return res;
    };
    static bool RegisterCreator(std::string const &k, std::function<TObj *(Args const &...)> const &fun) noexcept {
        return SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(k, fun).second;
    };
    template <typename U>
    static bool RegisterCreator(std::string const &k_hint = "") noexcept {
        return RegisterCreator(k_hint != "" ? k_hint : U::RegisterName(),
                               [](Args const &... args) { return new U(args...); });
    };
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::string const &k, U const &... args) {
        if (k == "") { return nullptr; }
        if (k.find("://") != std::string::npos) { return Create(std::make_shared<data::DataTable>(k), args...); }
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        std::shared_ptr<TObj> res = nullptr;
        auto it = f.find(k);

        if (it != f.end()) {
            res.reset(it->second(args...));
            LOGGER << TObj::RegisterName() << "::" << it->first << "  is created!" << std::endl;
        } else {
            std::ostringstream os;
            os << "Can not find Creator " << k << std::endl;
            os << std::endl << "Register " << TObj::RegisterName() << " Creator:" << std::endl;
            for (auto const &item : f) { os << item.first << std::endl; }
            WARNING << os.str();
        }
        return res;
    }
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::shared_ptr<DataEntity> const &cfg, U const &... args) {
        std::shared_ptr<TObj> res = nullptr;
        if (cfg == nullptr) {
        } else if (cfg->isLight()) {
            res = Create(data::DataCastTraits<std::string>::Get(cfg), args...);
        } else if (cfg->isTable()) {
            auto t = std::dynamic_pointer_cast<data::DataTable>(cfg);
            res = Create(t->GetValue<std::string>("Type", ""), args...);
            if (res != nullptr) res->Deserialize(t);
        }
        return res;
    }
};

#define DECLARE_REGISTER_NAME(_REGISTER_NAME_)                              \
   public:                                                                  \
    std::string GetRegisterName() const override { return RegisterName(); } \
    static std::string RegisterName() { return _REGISTER_NAME_; }           \
    static bool is_registered;

#define REGISTER_CREATOR(_CLASS_NAME_) bool _CLASS_NAME_::is_registered = _CLASS_NAME_::RegisterCreator<_CLASS_NAME_>();

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_ENABLECREATEFROMDATATABLE_H
