/**
 * @file line_cliping.h
 *
 * @date 2015-5-13
 * @author salmon
 */

#ifndef CORE_NUMERIC_LINE_CLIPING_H_
#define CORE_NUMERIC_LINE_CLIPING_H_

namespace simpla
{
/**
 *
 *
 *
 * @ref Maillot, Patrick-Gilles, “A New, Fast Method for 2D Polygon Clipping: Analysis and Software
 * Implementation,” ACM Trans. on Graphics, 11(3), July 1992, 276–290.
 * implement
 * @book Computer Graphics and Geometric Modelling ,Springer 2005 p.p.91
 */

class polygon_clip
{
	/* Constants */
	size_t MAXSIZE = 1000; /* maximum size of pnt2d array */
	size_t NOSEGM = 0; /* segment was rejected */
	size_t SEGM = 1; /* segment is at least partially visible */
	size_t CLIP = 2; /* segment was clipped */
	size_t TWOBITS = 10; /* flag for 2-bit code */

	/* Two lookup tables for finding turning point.
	 Tcc is used to compute a correct offset.
	 Cra gives an index into the clipRegion array for turning point coordinates. */
	size_t Tcc[15] =
	{ 0, -3, -6, 1, 3, 0, 1, 0, 6, 1, 0, 0, 1, 0, 0, 0 };
	size_t Cra[15] =
	{ -1, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1, -1, -1 };

	struct pnt2d
	{
		double x, y;
	};

	typedef pnt2d pnt2ds[MAXSIZE]; /* Global variables */

	/* The clipping region [xmin,xmax]X[ymin,ymax] bounds listed in order:
	 (xmin,ymin),(xmax,ymin),(xmin,ymax),(xmax,ymax) */

	pnt2d clipRegion[3];
	pnt2d startPt; /* start point of segment */
	size_t startC; /* code for start point */
	size_t startC0;/* saves startC for next call to CS_EndClip */
	pnt2d endPt; /*end point of segment */
	size_t endC; /* code for end point */
	size_t aC; /* used by procedure TwoBitEndPoint */

	void M_Clip(pnt2ds &inpts, size_t numin, pnt2ds & outpts, size_t &numout)
	/* inpts[0..numin-1] defines the input polygon with inpts[numin-1] = inpts[0] .
	 The clipped polygon is returned in outpts[0..numout-1]. It is assumed that
	 the array outpts is big enough. */
	{
		size_t i;
		numout = 0;
		/* Compute status of first point. If it is visible, it is stored in outpts array. */
		if (CS_StartClip() > 0)
		{
			outpts[numout] = startPt;
			++numout;
		}
		/* Now the rest of the points */
		for (i = 1; i < numin - 1; ++i)
		{
			size_t cflag = CS_EndClip(i);

			startC0 = endC; /* endC may get changed */

			if (((cflag & SEGM) != 0))
			{ /* Actually, this function should return true only if(the first point is clipped;
			 otherwise we generate redundant points. */
				if (((cflag & CLIP) != 0))
				{
					outpts[numout] = startPt;
					++numout;
				}
				outpts[numout] = endPt;
				++numout;
			}
			else if (((endC & TWOBITS) != 0))
			{
				/* The line has been rejected and we have a 2-bit end point. */
				if ((startC & endC & (TWOBITS - 1)) == 0)
				{
					/* The points have no region bits in common. We need to generate
					 an extra turning point - which one is specified by Cra table. */
					if ((startC & TWOBITS) != 0)
					{
						/* defines aC for this case */
						// BothAreTwoBits();
						/* Determines what aC should be by doing midpoint subdivision. */

						bool notdone;
						pnt2d Pt1, Pt2, aPt;
						notdone = true;
						Pt1 = startPt;
						Pt2 = endPt;

						while (notdone)
						{
							aPt.x = (Pt1.x + Pt2.x) / 2.0;
							aPt.y = (Pt1.y + Pt2.y) / 2.0;
							aC = ExtendedCsCode(aPt);
							if (((aC & TWOBITS) != 0))
							{
								if (aC == endC)
								{
									Pt2 = aPt;
								}
								else
								{
									if (aC == startC)
									{
										Pt1 = aPt;
									}
									else
									{
										notdone = false;
									}
								}
							}
							else
							{
								if ((aC & endC) != 0)
								{
									aC = endC + Tcc[startC & ~(TWOBITS)];
								}
								else
								{
									aC = startC + Tcc[endC & ~(TWOBITS)];
								}
							}
							notdone = false;
						}

					}
					else
					{
						aC = endC + Tcc[startC]; /* 1-bit start point, 2-bit end point */
					}

					outpts[numout] = clipRegion[Cra[aC & ~(TWOBITS)]];
					(++numout);
				}
			}
			else
			{

				/* The line has been rejected and we have a 1-bit end point. */

				if ((startC & TWOBITS) != 0)
				{
					if ((startC & endC) == 0)
						endC = startC + Tcc[endC];
				}
				else
				{
					endC = endC | startC;
					if (Tcc[endC] == 1)
					{
						endC = endC | TWOBITS;
					}
				}

			}

			/* The basic turning point test */
			if ((endC & TWOBITS) != 0)
			{
				outpts[numout] = clipRegion[Cra[endC & (~TWOBITS)]];
				++numout;
			}
			startPt = inpts[i];
		}

		/* Now close the output */
		if (numout > 0)
		{
			outpts[numout] = outpts[0];
			++numout;
		}
	} /* M_Clip */

	size_t ExtendedCsCode(pnt2d p)
	/* The Maillot extension of the Cohen-Sutherland encoding of points */
	{
		if (p.x < clipRegion[0].x)
		{
			if (p.y > clipRegion[3].y)
			{
				return (6 | TWOBITS);
			}
			if (p.y < clipRegion[0].y)
			{
				return (12 | TWOBITS);
			}
			return (4);
		}
		if (p.x > clipRegion[3].x)
		{
			if (p.y > clipRegion[3].y)
			{
				return (3 | TWOBITS);
			}
			if (p.y < clipRegion[0].y)
			{
				return (9 | TWOBITS);
			}
			return (1);
		}
		if (p.y > clipRegion[3].y)
		{
			return (2);
		}
		if (p.y < clipRegion[0].y)
		{
			return (8);
		}
		return (0);
	}
}
;
}  // namespace simpla

#endif /* CORE_NUMERIC_LINE_CLIPING_H_ */
