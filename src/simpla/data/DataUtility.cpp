//
// Created by salmon on 17-3-9.
//
#include "DataUtility.h"
#include "../../../to_delete/MPIAuxFunctions.h"
#include <simpla/parallel/MPIComm.h>
#include <simpla/utilities/Logo.h>
#include <simpla/utilities/MiscUtilities.h>
#include <simpla/utilities/parse_command_line.h>
#include "DataArray.h"
#include "DataBlock.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
//
// std::regex sub_group_regex=std::regex (R"(/([^/?#]+))", std::regex::optimize);
// std::regex match_path_regex=std::regex (R"(^(/?([/\S]+/)*)?([^/]+)?$)", std::regex::optimize);

std::shared_ptr<DataTable> ParseCommandLine(int argc, char **argv) {
    auto res = std::make_shared<DataTable>();

    parse_cmd_line(argc, argv, [&](std::string const &opt, std::string const &value) -> int {
        res->SetValue(opt, value);
        return CONTINUE;
    });
    return res;
};

void PackLua(std::shared_ptr<DataEntity> const &d, std::ostream &os, int indent = 0) {
    if (d == nullptr) {
    } else if (d->isTable()) {
        DataTable const &t = d->cast_as<DataTable>();

        os << "{";
        t.Foreach([&](std::string const &k, std::shared_ptr<DataEntity> const &v) {
            os << std::endl << std::setw(indent + 1) << " " << k << "= ";
            PackLua(v, os, indent + 1);
            os << ",";
        });

        os << std::endl
           << std::setw(indent) << " "
           << "}";

        //        os << "{";
        //        t.Foreach([&](std::string const &k, std::shared_ptr<DataEntity> const &v) {
        //            os << k << "= ";
        //            PackLua(v, os);
        //            os << "," << std::endl;
        //        });
        //        os << "}" << std::endl;
    } else if (d->isArray()) {
        auto const &t = d->cast_as<DataArray>();
        size_type num = t.size();
        os << "{";
        PackLua(t.Get(0), os, indent + 1);
        for (int i = 1; i < num; ++i) {
            os << ",";
            PackLua(t.Get(i), os, indent + 1);
        }
        os << "}";
    } else if (d->value_type_info() == typeid(bool)) {
        os << (DataCastTraits<bool>::Get(d) ? "true" : "false");
    } else if (d->isBlock()) {
        auto const &blk = d->cast_as<DataBlock>();
        int ndims = blk.GetNDIMS();
        os << "\"{ Dimensions = { {" << blk.GetInnerLowerIndex(0)[0];
        for (int i = 1; i < ndims; ++i) { os << "x" << blk.GetInnerLowerIndex(0)[i]; }
        os << "} , {" << blk.GetInnerUpperIndex(0)[0];
        for (int i = 1; i < ndims; ++i) { os << "x" << blk.GetInnerUpperIndex(0)[i]; }
        os << "}}}\"";
    } else {
        d->Serialize(os, 0);
    }
}

void Pack(std::shared_ptr<DataEntity> const &d, std::ostream &os, std::string const &type, int indent) {
    if (type == "lua") {
        PackLua(d, os, indent);
    } else {
        UNIMPLEMENTED;
    }
}

std::string AutoIncreaseFileName(std::string filename, std::string const &ext_str) {
    if (GLOBAL_COMM.process_num() == 0) {
        std::string prefix = filename;
        if (filename.size() > ext_str.size() && filename.substr(filename.size() - ext_str.size()) == ext_str) {
            prefix = filename.substr(0, filename.size() - ext_str.size());
        }

        /// @todo auto mkdir directory

        filename = prefix + AutoIncrease(
                                [&](std::string const &suffix) -> bool {
                                    std::string f = (prefix + suffix);
                                    return f == "" || *(f.rbegin()) == '/' || (CheckFileExists(f + ext_str));
                                },
                                0, 5) +
                   ext_str;
    }

    parallel::bcast_string(&filename);

    return filename;
}

}  // namespace data{
}  // namespace simpla{