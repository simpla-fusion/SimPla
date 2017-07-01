//
// Created by salmon on 17-4-12.
//
#include "Serializable.h"
#include "DataTable.h"
namespace simpla {
namespace data {
// Serializable::Serializable(){};
// Serializable::~Serializable() {}
std::shared_ptr<DataTable> Serializable::Pack() const { return std::make_shared<DataTable>(); };
void Serializable::Unpack(const std::shared_ptr<DataTable> &) {}
std::ostream &Serializable::Serialize(std::ostream &os, int indent) const { return Pack()->Serialize(os, indent); }
std::istream &Serializable::Deserialize(std::istream &is) { return is; }
std::ostream &operator<<(std::ostream &os, Serializable const &obj) { return obj.Serialize(os, 0); }
std::istream &operator>>(std::istream &is, Serializable &obj) { return obj.Deserialize(is); }
}  // namespace data{
}  // namespace simpla{