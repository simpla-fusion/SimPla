//
// Created by salmon on 17-11-17.
//

#include "Serializable.h"
#include "Configurable.h"
#include "DataEntry.h"
namespace simpla {
namespace data {
Serializable::Serializable() = default;
Serializable::Serializable(Serializable const &) = default;
Serializable::~Serializable() = default;
void Serializable::Deserialize(std::shared_ptr<const Entry> const &cfg) {
    if (auto *config = dynamic_cast<Configurable *>(this)) { config->Push(cfg); }
}
std::shared_ptr<Entry> Serializable::Serialize() const {
    auto res = Entry::New(Entry::DN_TABLE);
    res->SetValue("_TYPE_", FancyTypeName());
    if (auto const *config = dynamic_cast<const Configurable *>(this)) { config->Pop(res); }
    return res;
};

std::ostream &operator<<(std::ostream &os, Serializable const &obj) {
    os << *obj.Serialize();
    return os;
}
std::istream &operator>>(std::istream &is, Serializable &obj) {
    auto db = Entry::New(Entry::DN_TABLE);
    is >> *db;
    obj.Deserialize(db);
    return is;
}
}  // namespace geometry
}  // namespace simpla