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
    Serializable(){};
    virtual ~Serializable() {}
    virtual std::shared_ptr<DataTable> Serialize() const { return nullptr; };
    virtual void Deserialize(std::shared_ptr<DataTable> const &) {}
    virtual std::ostream &Serialize(std::ostream &os, int indent = 0) const { return os; }
    virtual std::istream &Deserialize(std::istream &is) { return is; }
};
inline std::ostream &operator<<(std::ostream &os, Serializable const &obj) {
    obj.Serialize(os, 0);
    return os;
}
inline std::istream &operator>>(std::istream &is, Serializable &obj) {
    obj.Deserialize(is);
    return is;
}

template <typename U>
std::shared_ptr<DataTable> const &Serialize(U const &u, ENABLE_IF((std::is_base_of<Serializable, U>::value))) {
    return u.Serialize();
}

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_SERIALIZABLE_H
