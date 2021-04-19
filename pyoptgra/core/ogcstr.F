      SUBROUTINE OGCSTR (STRCON, LENCON)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE CONSTRAINT + MERIT STRING
C ======================================================================
C INP | STRCON(NUMCON+1) | C80 | CONIABLES NAME STRING
C ----------------------------------------------------------------------
C INP | LENCON(NUMCON+1) | I*4 | CONIABLES NAME LENGTH
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      CHARACTER*80 STRCON(NUMCON+1)
      INTEGER(4)   LENCON(NUMCON+1)
C ======================================================================
      INTEGER(4)   CON, LEN
C ======================================================================
      DO CON = 1, NUMCON + 1
         LEN = MIN (LENCON(CON), 80)
         CONSTR(CON) = STRCON(CON)
         CONLEN(CON) = LEN
      ENDDO
C ======================================================================
      RETURN
      END
