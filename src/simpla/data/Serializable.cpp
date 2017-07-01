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
std::ostream &Serializable::Pack(std::ostream &os, int indent) const { return Pack()->Pack(os, indent); }
std::istream &Serializable::Unpack(std::istream &is) { return is; }
std::ostream &operator<<(std::ostream &os, Serializable const &obj) { return obj.Pack(os, 0); }
std::istream &operator>>(std::istream &is, Serializable &obj) { return obj.Unpack(is); }
}  // namespace data{
}  // namespace simpla{