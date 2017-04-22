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
template <typename TObj>
class EnableCreateFromDataTable {
   public:
    EnableCreateFromDataTable() = default;
    virtual ~EnableCreateFromDataTable() = default;

    struct ObjectFactory {
        std::map<std::string, std::pair<std::function<TObj *()>, std::string>> m_factory_;
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
    static bool RegisterCreator(std::string const &k, std::function<TObj *()> const &fun,
                                std::string const &desc_s = "") noexcept {
        return SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(k, std::make_pair(fun, desc_s)).second;
    };
    template <typename U>
    static bool RegisterCreator(std::string const &k, std::string const &desc_s = "") noexcept {
        return RegisterCreator(k, []() { return new U; }, desc_s);
    };

    static std::shared_ptr<TObj> Create(std::string const &k) {
        auto const &f = SingletonHolder<ObjectFactory>::instance().m_factory_;
        std::shared_ptr<TObj> res = nullptr;
        auto it = f.find(k);

        if (it != f.end()) {
            res.reset(it->second.first());
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
    static std::shared_ptr<TObj> Create(std::shared_ptr<DataTable> const &cfg) {
        if (cfg == nullptr) { return nullptr; }
        auto res = Create(cfg->GetValue<std::string>("Type", "unnamed"));
        res->Deserialize(cfg);
        return res;
    }
};
#define REGISTER_CREATOR(_BASE_CLASS_NAME_, _CLASS_NAME_, _DESC_) \
    bool _CLASS_NAME_##_IS_REGISTERED_ =                          \
        _BASE_CLASS_NAME_::RegisterCreator<_CLASS_NAME_>(__STRING(_CLASS_NAME_), _DESC_);

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_ENABLECREATEFROMDATATABLE_H
