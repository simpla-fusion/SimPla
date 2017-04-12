//
// Created by salmon on 17-4-12.
//

#ifndef SIMPLA_ENABLECREATEFROMDATATABLE_H
#define SIMPLA_ENABLECREATEFROMDATATABLE_H

#include <iostream>
#include <memory>
#include <string>
#include "DataTable.h"
#include "simpla/concept/CheckConcept.h"
#include "simpla/design_pattern/SingletonHolder.h"
namespace simpla {
namespace data {
class DataTable;
template <typename TObj>
class EnableCreateFromDataTable {
   public:
    EnableCreateFromDataTable(){};
    virtual ~EnableCreateFromDataTable() {}

    struct ObjectFactory {
        std::map<std::string, std::function<TObj *()>> m_factory_;
    };
    static bool RegisterCreator(std::string const &k, std::function<TObj *()> const &fun) {
        return SingletonHolder<ObjectFactory>::instance().m_factory_.emplace(k, fun).second;
    };
    template <typename U>
    static bool RegisterCreator(std::string const &k) {
        return RegisterCreator(k, []() { return new U; });
    };

    static std::shared_ptr<TObj> Create(std::string const &k) {
        std::shared_ptr<TObj> res = nullptr;
        if (k == "") {
            res = std::make_shared<TObj>();
        } else {
            try {
                res.reset(SingletonHolder<ObjectFactory>::instance().m_factory_.at(k)());
            } catch (std::out_of_range const &) {
                std::ostringstream oss;
                oss <<  TObj::ClassName() << "::" << k << " is not registered. [ ";
                for (auto const &item : SingletonHolder<ObjectFactory>::instance().m_factory_) {
                    oss << item.first << ",";
                }
                oss << "]" << std::endl;
                RUNTIME_ERROR << oss.str() << std::endl;
            }
        }
        if (res != nullptr) { LOGGER << TObj::ClassName() << "::" << k << "  is created!" << std::endl; }
        return res;
    }
    static std::shared_ptr<TObj> Create(std::shared_ptr<DataTable> const &cfg) {
        if (cfg == nullptr) { return nullptr; }
        auto res = Create(cfg->GetValue<std::string>("Type", "unnamed"));
        res->Deserialize(cfg);
        return res;
    }
};
#define REGISTER_CREATOR(_BASE_CLASS_NAME_, _CLASS_NAME_) \
    bool _CLASS_NAME_##_IS_REGISTERED_ = _BASE_CLASS_NAME_::RegisterCreator<_CLASS_NAME_>(__STRING(_CLASS_NAME_));

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_ENABLECREATEFROMDATATABLE_H
