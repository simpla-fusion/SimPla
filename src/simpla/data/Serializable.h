//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include "DataEntry.h"
namespace simpla {
namespace data {
struct Serializable {
   public:
    Serializable();
    Serializable(Serializable const &);
    ~Serializable();
    virtual std::string FancyTypeName() const { return "unknown"; }
    virtual void Deserialize(std::shared_ptr<const DataEntry> const &cfg);
    virtual std::shared_ptr<DataEntry> Serialize() const;
    template <typename TObj>
    static std::shared_ptr<TObj> Create(std::shared_ptr<DataEntry> const &cfg) {
        auto res = TObj::Create(cfg->GetValue("_TYPE_", ""));
        res->Deserialize(cfg);
        return res;
    }
};
std::ostream &operator<<(std::ostream &os, Serializable const &obj);
std::istream &operator>>(std::istream &is, Serializable &obj);

}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SERIALIZABLE_H
