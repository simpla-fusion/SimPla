//
// Created by salmon on 17-4-12.
//

#ifndef SIMPLA_ENABLECREATEFROMDATATABLE_H
#define SIMPLA_ENABLECREATEFROMDATATABLE_H


#include <iostream>
#include <memory>
#include <string>
#include "simpla/concept/CheckConcept.h"
#include "simpla/design_pattern/SingletonHolder.h"

namespace simpla{
namespace data{

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
        try {
            res.reset(SingletonHolder<ObjectFactory>::instance().m_factory_.at(k)());
        } catch (std::out_of_range const &) { RUNTIME_ERROR << "Missing object creator  [" << k << "]!" << std::endl; }

        if (res != nullptr) { LOGGER << "Object [" << k << "] is created!" << std::endl; }
        return res;
    }
    static std::shared_ptr<TObj> Create(std::shared_ptr<DataTable> const &cfg) {
        if (cfg == nullptr) { return nullptr; }
        auto res = Create(cfg->GetValue<std::string>("Type", "unnamed"));
        res->Deserialize(cfg);
        return res;
    }
};
template <typename U>
std::shared_ptr<U> Deserialize(std::shared_ptr<DataTable> const &d,
                               ENABLE_IF((std::is_base_of<EnableCreateFromDataTable<U>, U>::value))) {
return U::Create(d);
}
}//namespace data{

}//namespace simpla{
#endif //SIMPLA_ENABLECREATEFROMDATATABLE_H
