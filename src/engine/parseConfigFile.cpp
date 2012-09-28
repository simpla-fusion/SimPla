/*
 * parserConfigFile.cpp
 *
 *  Created on: 2011-12-22
 *      Author: salmon
 */
#include "context.h"
#include "engine/engine.h"

#include "emfield/emfield.h"
#include "fluid/fluid.h"
#include "pic/pic.h"
#include "io/io.h"
#include "utilities/luaParser.h"

Context::Holder parseConfigFile(std::string const &cFile)
{
	LuaParser cfgPaser;
	Context::Holder self(
			Context::create(cfgPaser.toValue<std::string>("UNIT_DIMENSIONS")));

	cfgPaser.parseFile(cFile);

	cfgPaser.getValue("DT", self->grid->dt);

	cfgPaser.getValue("DIMS", self->grid->dims);

	cfgPaser.getValue("GW", self->grid->ghostWidth);

	self->grid->xmin = 0;

	self->grid->xmax = self->grid->xmin + cfgPaser.toValue<Vec3>("LENGTH");

	cfgPaser.getValue("BC", self->bc);

	cfgPaser.getValue("SPECIES", self->species);

	cfgPaser.getValue("DESCRIPTION", self->desc);

	// load fields and particles  ====================================

	std::list < std::pair<std::string, int> > loadField;

	cfgPaser.getValue("LOAD_FIELDS", loadField);

	for (std::list<std::pair<std::string, int> >::iterator it =
			loadField.begin(); it != loadField.end(); ++it)
	{

		switch (it->second)
		{

		case IZeroForm:
			self->registerFunction(
					boost::bind(&LuaParser::fillArray2<Context::ZeroForm>,
							cfgPaser, it->first,
							self->getField<IZeroForm, Real>(it->first)),
					"Load field" + it->first, -1);
			break;
		case IVecZeroForm:
			self->registerFunction(
					boost::bind(&LuaParser::fillArray2<Context::VecZeroForm>,
							cfgPaser, it->first,
							self->getField<IVecZeroForm, Real>(it->first)),
					"Load field" + it->first, -1);
			break;
		case IOneForm:
			self->registerFunction(
					boost::bind(&LuaParser::fillArray2<Context::OneForm>,
							cfgPaser, it->first,
							self->getField<IOneForm, Real>(it->first)),
					"Load field" + it->first, -1);
			break;
		case ITwoForm:
			self->registerFunction(
					boost::bind(&LuaParser::fillArray2<Context::TwoForm>,
							cfgPaser, it->first,
							self->getField<ITwoForm, Real>(it->first)),
					"Load field" + it->first, -1);
			break;
		}
	}

	cfgPaser.getValue("SP_LIST", self->species);

	// register functions  ====================================

	LocalComm::Holder comm = LocalComm::create(self);

	self->communicateField = boost::bind(&LocalComm::updateField, comm, _1);

	self->communicateParticle = boost::bind(&LocalComm::updateParticle, comm,
			_1);

	// set current source  ====================================

//	Context::OneForm::Holder Ji = self->getField<IOneForm, Real>("Ji");
//
//	if (cfgPaser.check("JSrc")) {
//
//		self->registerFunction(boost::bind(&Context::OneForm::clear, self->getField<IOneForm, Real>("Ji")), "clear J1",
//				Context::PROCESS);
//
//		self->registerFunction(boost::bind(&LuaParser::getExprToArray<Context::OneForm>, cfgPaser,
////						(bf("JSrc(%e)") % getTime()).str(),
//				std::string("JSrc(0)"), self->getField<IOneForm, Real>("Ji")), "External current (J) source",
//				Context::PROCESS);
//	}

// set engine functions  ====================================

	PIC::registerFunction(self);

	Fluid::registerFunction(self);

	em_field::registerFunction(self);

	return self;

}

