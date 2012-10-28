/*
 * Write.cpp
 *
 *  Created on: 2011-12-10
 *      Author: salmon
 */

#include "io.h"
#include "write_hdf5.h"
#include "write_xdmf.h"
namespace IO
{

void registerFunction(Context::Holder ctx,
		std::vector<std::string> const & diag_list, const std::string & output,
		std::string const & format, SizeType record)
{

	if (format == "HDF5")
	{

		IO::WriteHDF5::Holder holder = IO::WriteHDF5::create(ctx, record,
				output);

		ctx->registerFunction(TR1::bind(&IO::WriteHDF5::pre_process, holder),
				" Open HDF5 file " + output, -1);

		for (std::vector<std::string>::const_iterator it = diag_list.begin();
				it != diag_list.end(); ++it)
		{

			ctx->registerFunction(
					TR1::bind(&IO::WriteHDF5::writeField, holder, *it),
					" Write field [" + *it + "] to file " + output, 0);

		}
		ctx->registerFunction(TR1::bind(&IO::WriteHDF5::post_process, holder),
				" Close HDF5 file " + output, 1);

	}
	else if (format == "XDMF")
	{

		IO::WriteXDMF::Holder holder = IO::WriteXDMF::create(ctx, record,
				output);

		ctx->registerFunction(TR1::bind(&IO::WriteXDMF::writeXDMFHead, holder),
				"Write XDMF head", 0);
		for (std::vector<std::string>::const_iterator it = diag_list.begin();
				it != diag_list.end(); ++it)
		{

			ctx->registerFunction(
					TR1::bind(&IO::WriteXDMF::writeField, holder, *it),
					"  Write field[" + *it + "] to file " + output, 0);

		}
		ctx->registerFunction(TR1::bind(&IO::WriteXDMF::writeXDMFFoot, holder),
				"Write XDMF foot", 0);
	}

}

} /* namespace IO */
