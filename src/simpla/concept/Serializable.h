//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include "CheckConcept.h"
#include "simpla/data/DataTable.h"
#include "simpla/design_pattern/SingletonHolder.h"
namespace simpla {
namespace concept {

template <typename TObj>
class Serializable {
   public:
    Serializable(){};
    virtual ~Serializable() {}
    virtual std::shared_ptr<data::DataTable> Serialize() const { return std::make_shared<data::DataTable>(); };
    virtual void Deserialize(std::shared_ptr<data::DataTable> const &) {}

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
    static std::shared_ptr<TObj> Create(std::shared_ptr<data::DataTable> const &cfg) {
        if (cfg == nullptr) { return nullptr; }
        auto res = Create(cfg->GetValue<std::string>("Type", "unnamed"));
        res->Deserialize(cfg);
        return res;
    }
};

}  // namespace concept{

template <typename U>
std::shared_ptr<data::DataTable> const &Serialize(U const &u,
                                                  ENABLE_IF((std::is_base_of<concept::Serializable<U>, U>::value))) {
    return u.Serialize();
}
template <typename U>
std::shared_ptr<U> Deserialize(std::shared_ptr<data::DataTable> const &d,
                               ENABLE_IF((std::is_base_of<concept::Serializable<U>, U>::value))) {
    return std::shared_ptr<U>(concept::Serializable<U>::Create(d));
}

}  // namespace simpla{
#endif  // SIMPLA_SERIALIZABLE_H
