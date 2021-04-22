      SUBROUTINE OGDERI (DERVAR, PERVAR)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE COMPUTATION OF DERIVATIVES
C ======================================================================
C INP | DERVAR           | I*4 | DERIVATIVES COMPUTATION MODE
C     |                  |     | -> 1: USER DEFINED
C     |                  |     | -> 2: NUMERIC WITH DOUBLE DIFFERENCING
C     |                  |     | -> 3: NUMERIC WITH SINGLE DIFFERENCING
C ----------------------------------------------------------------------
C INP | PERVAR(NUMVAR)   | R*8 | VARIABLES PERTURBATION FOR DERIVATIVES
C     |                  |     | -> NOT SCALED
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   DERVAR
      REAL(8)      PERVAR(NUMVAR)
C ======================================================================
      INTEGER(4)   VAR
C ======================================================================
      VARDER = DERVAR
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         VARPER(VAR) = PERVAR(VAR)
      ENDDO
C ======================================================================
      RETURN
      END
