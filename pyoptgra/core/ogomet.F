      SUBROUTINE OGOMET (METOPT)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE OPTIMISATION CONTROL PARAMETERS
C ======================================================================
C INP | METOPT | I*4 | OPTIMISATION METHOD
C     |        |     | 3: CONJUGATE GRADIENT METHOD
C     |        |     | 2: SPETRAL CONJUGATE GRADIENT METHOD
C     |        |     | 1: MODIFIED SPETRAL CONJUGATE GRADIENT METHOD
C     |        |     | 0: STEEPEST DESCENT METHOD
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   METOPT
C ======================================================================
      OPTMET = METOPT
C ======================================================================
      RETURN
      END
