//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include <iostream>
#include <memory>
#include <string>
namespace simpla {
namespace data {
class DataTable;

class Serializable {
   public:
    Serializable();
    virtual ~Serializable();
    virtual std::shared_ptr<DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<DataTable>);
    virtual std::ostream &Serialize(std::ostream &os, int indent = 0) const;
    virtual std::istream &Deserialize(std::istream &is);
};
std::ostream &operator<<(std::ostream &os, Serializable const &obj);
std::istream &operator>>(std::istream &is, Serializable &obj);

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_SERIALIZABLE_H
