//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_SAMRAIDATABASE_H
#define SIMPLA_SAMRAIDATABASE_H

#include <SAMRAI/tbox/Database.h>
#include <vector>
#include <string>

namespace simpla
{

class SAMRAIDataBase : public SAMRAI::tbox::Database
{
    SAMRAIDataBase() {};


    virtual ~SAMRAIDataBase() {};


    virtual bool create(const std::string &name) { m_data_ = m_data_->create(); };


    virtual bool open(const std::string &name, const bool read_write_mode = false)
    {
        UNIMPLEMENTED;
        return false;
    };


    virtual bool close()
    {
        UNIMPLEMENTED;
        return false;
    };

    virtual bool keyExists(const std::string &key) { return m_data_->has(key); };


    virtual std::vector<std::string> getAllKeys() = 0;


    virtual enum DataType getArrayType(const std::string &key) = 0;


    virtual size_t getArraySize(const std::string &key) = 0;


    virtual bool isDatabase(const std::string &key) = 0;


    virtual boost::shared_ptr<Database> putDatabase(const std::string &key) = 0;

    virtual boost::shared_ptr<Database> getDatabase(const std::string &key)
    {
        return std::make_shared<SAMRAIDataBase>(m_data_->get(key));
    };


    virtual boost::shared_ptr<Database>
    getDatabaseWithDefault(
            const std::string &key,
            const boost::shared_ptr<Database> &defaultvalue);


    virtual bool
    isBool(
            const std::string &key) = 0;


    virtual void
    putBool(
            const std::string &key,
            const bool &data);


    virtual void
    putBoolVector(
            const std::string &key,
            const std::vector<bool> &data);


    virtual void
    putBoolArray(
            const std::string &key,
            const bool *const data,
            const size_t nelements) = 0;


    virtual bool
    getBool(
            const std::string &key);


    virtual bool
    getBoolWithDefault(
            const std::string &key,
            const bool &defaultvalue);


    virtual std::vector<bool>
    getBoolVector(
            const std::string &key) = 0;

    virtual void
    getBoolArray(
            const std::string &key,
            bool *data,
            const size_t nelements);


    virtual bool
    isDatabaseBox(
            const std::string &key) = 0;


    virtual void
    putDatabaseBox(
            const std::string &key,
            const DatabaseBox &data);


    virtual void
    putDatabaseBoxVector(
            const std::string &key,
            const std::vector<DatabaseBox> &data);


    virtual void
    putDatabaseBoxArray(
            const std::string &key,
            const DatabaseBox *const data,
            const size_t nelements) = 0;


    virtual DatabaseBox
    getDatabaseBox(
            const std::string &key);


    virtual DatabaseBox
    getDatabaseBoxWithDefault(
            const std::string &key,
            const DatabaseBox &defaultvalue);


    virtual std::vector<DatabaseBox>
    getDatabaseBoxVector(
            const std::string &key) = 0;


    virtual void
    getDatabaseBoxArray(
            const std::string &key,
            DatabaseBox *data,
            const size_t nelements);


    virtual bool
    isChar(
            const std::string &key) = 0;


    virtual void
    putChar(
            const std::string &key,
            const char &data);


    virtual void
    putCharVector(
            const std::string &key,
            const std::vector<char> &data);


    virtual void
    putCharArray(
            const std::string &key,
            const char *const data,
            const size_t nelements) = 0;


    virtual char
    getChar(
            const std::string &key);


    virtual char
    getCharWithDefault(
            const std::string &key,
            const char &defaultvalue);


    virtual std::vector<char>
    getCharVector(
            const std::string &key) = 0;


    virtual void
    getCharArray(
            const std::string &key,
            char *data,
            const size_t nelements);


    virtual bool
    isComplex(
            const std::string &key) = 0;


    virtual void
    putComplex(
            const std::string &key,
            const dcomplex &data);


    virtual void
    putComplexVector(
            const std::string &key,
            const std::vector<dcomplex> &data);


    virtual void
    putComplexArray(
            const std::string &key,
            const dcomplex *const data,
            const size_t nelements) = 0;


    virtual dcomplex
    getComplex(
            const std::string &key);


    virtual dcomplex
    getComplexWithDefault(
            const std::string &key,
            const dcomplex &defaultvalue);


    virtual std::vector<dcomplex>
    getComplexVector(
            const std::string &key) = 0;

    virtual void
    getComplexArray(
            const std::string &key,
            dcomplex *data,
            const size_t nelements);


    virtual bool
    isDouble(
            const std::string &key) = 0;


    virtual void
    putDouble(
            const std::string &key,
            const double &data);


    virtual void
    putDoubleVector(
            const std::string &key,
            const std::vector<double> &data);


    virtual void
    putDoubleArray(
            const std::string &key,
            const double *const data,
            const size_t nelements) = 0;


    virtual double
    getDouble(
            const std::string &key);


    virtual double
    getDoubleWithDefault(
            const std::string &key,
            const double &defaultvalue);


    virtual std::vector<double>
    getDoubleVector(
            const std::string &key) = 0;


    virtual void
    getDoubleArray(
            const std::string &key,
            double *data,
            const size_t nelements);


    virtual bool
    isFloat(
            const std::string &key) = 0;


    virtual void
    putFloat(
            const std::string &key,
            const float &data);


    virtual void
    putFloatVector(
            const std::string &key,
            const std::vector<float> &data);


    virtual void
    putFloatArray(
            const std::string &key,
            const float *const data,
            const size_t nelements) = 0;


    virtual float
    getFloat(
            const std::string &key);


    virtual float
    getFloatWithDefault(
            const std::string &key,
            const float &defaultvalue);


    virtual std::vector<float>
    getFloatVector(
            const std::string &key) = 0;


    virtual void
    getFloatArray(
            const std::string &key,
            float *data,
            const size_t nelements);


    virtual bool
    isInteger(
            const std::string &key) = 0;


    virtual void
    putInteger(
            const std::string &key,
            const int &data);


    virtual void
    putIntegerVector(
            const std::string &key,
            const std::vector<int> &data);


    virtual void
    putIntegerArray(
            const std::string &key,
            const int *const data,
            const size_t nelements) = 0;


    virtual int
    getInteger(
            const std::string &key);


    virtual int
    getIntegerWithDefault(
            const std::string &key,
            const int &defaultvalue);


    virtual std::vector<int>
    getIntegerVector(
            const std::string &key) = 0;


    virtual void
    getIntegerArray(
            const std::string &key,
            int *data,
            const size_t nelements);


    virtual bool
    isString(
            const std::string &key) = 0;


    virtual void
    putString(
            const std::string &key,
            const std::string &data);


    virtual void
    putStringVector(
            const std::string &key,
            const std::vector<std::string> &data);


    virtual void
    putStringArray(
            const std::string &key,
            const std::string *const data,
            const size_t nelements) = 0;


    virtual std::string
    getString(
            const std::string &key);


    virtual std::string
    getStringWithDefault(
            const std::string &key,
            const std::string &defaultvalue);


    virtual std::vector<std::string>
    getStringVector(
            const std::string &key) = 0;


    virtual void
    getStringArray(
            const std::string &key,
            std::string *data,
            const size_t nelements);


    virtual bool isVector(const std::string &key);

    virtual std::string getName() { return "unamed"; };

    virtual void printClassData(std::ostream &os = pout) { m_data_.print(os); };

    std::shard_ptr<toolbox::DataBase> m_data_;
};

}
#endif //SIMPLA_SAMRAIDATABASE_H
