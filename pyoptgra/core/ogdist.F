      SUBROUTINE OGDIST (MAXVAR, SNDVAR)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE OPTIMISATION CONTROL PARAMETERS
C ======================================================================
C INP | ITEMAX           | I*4 | MAXIMUM NUMBER OF ITERATIONS
C ----------------------------------------------------------------------
C INP | MAXVAR           | R*8 | MAXIMUM DISANCE PER ITERATION
C     |                  |     | -> SCALED
C ----------------------------------------------------------------------
C INP | SNDVAR           | R*8 | PERTURBATION FOR SND ORDER DERIVATIVES
C     |                  |     | -> SCALED
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      REAL(8)      MAXVAR
      REAL(8)      SNDVAR
C ======================================================================
      VARMAX = MAXVAR
      VARSND = SNDVAR
C ======================================================================
      RETURN
      END
