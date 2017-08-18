//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include <memory>
namespace simpla {
namespace data {
class DataEntity;
class DataEntry {
    virtual std::shared_ptr<DataEntity> GetParent() const { return nullptr; }
    virtual int GetNumberOfChild() const;
    virtual bool isLeaf() const;

    virtual std::shared_ptr<DataEntity> GetChild() const { return nullptr; }
    virtual std::shared_ptr<DataEntity> GetNext() const { return nullptr; }
    virtual int GetIndex() const { return 0; }
    virtual std::string GetName() const { return ""; }
};
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
