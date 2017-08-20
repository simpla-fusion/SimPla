//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include "simpla/SIMPLA_config.h"

#include <memory>
#include "DataEntity.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {
class DataEntity;
class DataNode : public std::enable_shared_from_this<DataNode> {
    SP_OBJECT_BASE(DataNode);

   protected:
    DataNode();

   public:
    virtual ~DataNode();
    SP_DEFAULT_CONSTRUCT(DataNode);

    static std::shared_ptr<DataNode> New(std::string const& k = "");

    /** @addtogroup{ capacity */
    virtual bool isNull() { return true; }
    /** @} */
    /** @addtogroup{ access */
    virtual std::shared_ptr<DataNode> Root() { return FindNode("/", true); }
    virtual std::shared_ptr<DataNode> Parent() const { return FindNode("..", true); }

    virtual std::shared_ptr<DataNode> FirstChild() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    virtual std::shared_ptr<DataNode> NewNode(std::string const& uri, bool recursive = false) { return nullptr; };

    virtual std::shared_ptr<DataNode> FindNode(std::string const& uri, bool recursive = false) const {
        return nullptr;
    };

    virtual size_type GetNumberOfChildren() const { return 0; }
    virtual std::shared_ptr<DataNode> GetNodeByIndex(index_type idx) const { return nullptr; }
    virtual std::shared_ptr<DataNode> GetNodeByName(std::string const& s) const { return nullptr; }
    /** @} */

    /** @addtogroup{  modify */
    virtual int SetNodeByIndex(index_type idx, std::shared_ptr<DataNode> const& v) { return 0; }
    virtual int SetNodeByName(std::string const& k, std::shared_ptr<DataNode> const& v) { return 0; }
    virtual std::shared_ptr<DataNode> AddNode(std::shared_ptr<DataNode> const& v = nullptr) { return nullptr; }
    /** @}  */

    /** @addtogroup{ */
    virtual std::shared_ptr<DataEntity> Value() const { return nullptr; }

    template <typename V>
    V GetValue(std::string const& k, bool recursive = true) const {
        auto p = FindNode(k, recursive);
        if (p == nullptr) { OUT_OF_RANGE; }
        return p->Value()->as<V>();
    }

    template <typename V>
    V GetValue(std::string const& k, V const& default_value, bool recursive = true) const {
        auto p = FindNode(k, recursive);
        return p == nullptr ? default_value : p->Value()->as<V>();
    }

    /** @} */
};

std::ostream& operator<<(std::ostream, DataNode const&);
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
