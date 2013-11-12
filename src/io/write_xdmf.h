/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$*/
#ifndef SRC_IO_WRITE_XDMF_H_
#define SRC_IO_WRITE_XDMF_H_

#include <H5CommonFG.h>
#include <H5File.h>
#include <H5Fpublic.h>
#include <H5Group.h>
//#include <H5public.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "../fetl/field.h"
#include "../fetl/ntuple_ops.h"
#include "../fetl/primitives.h"
#include "../mesh/uniform_rect.h"

namespace H5 { class FileAccPropList; }
namespace H5 { class FileCreatPropList; }

namespace simpla
{

template<typename TM>
class WriteXDMF
{

public:

	typedef TM mesh_type;
	typedef WriteXDMF<mesh_type> this_type;
	mesh_type const & mesh;

	WriteXDMF(mesh_type const &pmesh, std::string const & path) :
			mesh(pmesh), file_template_(CreateFileTemplate(pmesh)),

			attr_place_holder_("<!-- ADD_ATTRITUTE_HERE -->"),

			out_path_(path),

			counter_(0),

			xmdf_file_buffer_(file_template_)
	{
	}

	~WriteXDMF()
	{
	}

	template<int IFORM, typename TV>
	void AddAttribute(Field<Geometry<mesh_type, IFORM>, TV> const &f,
			std::string name)
	{

		typedef typename Field<Geometry<mesh_type, IFORM>, TV>::field_value_type field_value_type;

		std::string attr_str;

		std::ostringstream ss;
		size_t extents = f.size();

		if (is_ntuple<field_value_type>::value)
		{
			ss

			<< "  <Attribute Name='" << name << "' \n"

			<< "      AttributeType='Vector' Center='Node' >\n"

			<< "    <DataItem  NumberType='Float' Precision='8'  "

			<< "Format='HDF' Dimensions='" << extents << " "

			<< nTupleTraits<field_value_type>::NUM_OF_DIMS << "' > \n"

			<< filename_ << ".h5:/" << name

			<< "    </DataItem> \n"

			<< "  </Attribute>\n";
		}
		else
		{
			ss

			<< "  <Attribute Name='" << name << "' \n"

			<< "      AttributeType='Scalar' Center='Node' >\n"

			<< "    <DataItem  NumberType='Float' Precision='8'  "

			<< "Format='HDF' Dimensions='" << extents << "' > \n"

			<< filename_ << ".h5:/" << name

			<< "    </DataItem> \n"

			<< "  </Attribute>\n";
		}

		HDF5Write(h5_grp_, name, f);

		xmdf_file_buffer_.insert(xmdf_file_buffer_.find(attr_place_holder_, 0),
				ss.str());

	}

	void Flush()
	{
		std::fstream fs((out_path_ + "/" + filename_ + ".xdmf").c_str(),
				std::fstream::out);

		fs << xmdf_file_buffer_;

		fs.close();

		++counter_;

	}

	void Init()
	{
		xmdf_file_buffer_ = file_template_;

		h5_file_ = H5::H5File(out_path_ + "/" + filename_ + ".h5",
		H5F_ACC_TRUNC);
		h5_grp_ = h5_file_.openGroup("/");

	}

private:

	std::string file_template_;
	std::string attr_place_holder_;

	std::string out_path_;
	std::string filename_;
	std::string xmdf_file_buffer_;

	H5::H5File h5_file_;
	H5::Group h5_grp_;
	size_t counter_;

};

std::string CreateFileTemplate(UniformRectMesh const & mesh);

} // namespace simpla
#endif  // SRC_IO_WRITE_XDMF_H_
