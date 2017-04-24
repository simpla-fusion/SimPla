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

    virtual std::string GetClassName() const { return TObj::ClassName(); }

    struct ObjectFactory {
        std::map<std::string, std::pair<std::function<TObj *(Args &&...)>, std::string>> m_factory_;
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
            res = it->second.second;
        } else {
            std::ostringstream os;
            os << std::endl << "Register " << TObj::ClassName() << " Creator:" << std::endl;
            for (auto const &item : f) {
                os << std::setw(15) << item.first << " : " << item.second.second << std::endl;
            }
            res = os.str();
        }
        return res;
    };
    static bool RegisterCreator(std::string const &k, std::function<TObj *(Args &&...)> const &fun,
                                std::string const &desc_s = "") noexcept {
        return SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(k, std::make_pair(fun, desc_s)).second;
    };
    template <typename U>
    static bool RegisterCreator(std::string const &desc_s = "") noexcept {
        return RegisterCreator(U::ClassName(), [](Args const &... args) { return new U(args...); }, desc_s);
    };
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::string const &k, U &&... args) {
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        std::shared_ptr<TObj> res = nullptr;
        auto it = f.find(k);

        if (it != f.end()) {
            res.reset(it->second.first(std::forward<U>(args)...));
            LOGGER << TObj::ClassName() << "::" << it->first << "  is created!" << std::endl;
        } else {
            std::ostringstream os;
            os << "Can not find Creator " << k << std::endl;
            os << std::endl << "Register " << TObj::ClassName() << " Creator:" << std::endl;
            for (auto const &item : f) {
                os << std::setw(15) << item.first << " : " << item.second.second << std::endl;
            }
            RUNTIME_ERROR << os.str();
        }
        return res;
    }
    template <typename... U>
    static std::shared_ptr<TObj> Create(std::shared_ptr<DataTable> const &cfg, U &&... args) {
        if (cfg == nullptr) { return nullptr; }
        auto res = Create(cfg->GetValue<std::string>("Type", "unnamed"), std::forward<U>(args)...);
        res->Deserialize(cfg);
        return res;
    }
};

#define DECLARE_REGISTER_NAME(_CLASS_NAME_)                           \
   public:                                                            \
    std::string GetClassName() const override { return ClassName(); } \
    static std::string ClassName() { return _CLASS_NAME_; }           \
    static bool is_registered;

#define REGISTER_CREATOR(_CLASS_NAME_) bool _CLASS_NAME_::is_registered = _CLASS_NAME_::RegisterCreator<_CLASS_NAME_>();

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_ENABLECREATEFROMDATATABLE_H
