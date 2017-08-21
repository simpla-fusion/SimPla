//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATANODEXML_H
#define SIMPLA_DATANODEXML_H
#include "../DataNode.h"
namespace simpla {
namespace data {
class DataNodeXML : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeXML, DataNode);
    struct pimpls_s;
    pimpls_s* m_pimpl_ = nullptr;

   protected:
    DataNodeXML();

   public:
    //    enum TypeEnum { SP_DATA_NODE_NULL, SP_DATA_NODE_ENTITY, SP_DATA_NODE_ARRAY, SP_DATA_NODE_TABLE };

    virtual ~DataNodeXML();
    SP_DEFAULT_CONSTRUCT(DataNodeXML);

    static std::shared_ptr<DataNodeXML> New();

    /** @addtogroup{ capacity */
    virtual bool isNull() { return true; }
    /** @} */
    /** @addtogroup{ access */

    virtual std::shared_ptr<DataNode> FirstChild() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    virtual size_type GetNumberOfChildren() const { return 0; }
    virtual std::shared_ptr<DataNode> GetNodeByIndex(index_type idx) const { return nullptr; }
    virtual std::shared_ptr<DataNode> GetNodeByName(std::string const& s) const { return nullptr; }
    /** @} */

    /** @addtogroup{  modify */
    virtual int SetNodeByIndex(index_type idx, std::shared_ptr<DataNode> const& v) { return 0; }

    virtual int SetNodeByName(std::string const& k, std::shared_ptr<DataNode> const& v) { return 0; }
    virtual int AddNode(std::shared_ptr<DataNode> const& v) { return 0; }
    /** @}  */

    /** @addtogroup{ */
    virtual std::string Key() const;
    virtual std::shared_ptr<DataEntity> Value() const { return nullptr; }
};
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATANODEXML_H
