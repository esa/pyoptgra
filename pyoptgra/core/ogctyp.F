      SUBROUTINE OGCTYP (TYPCON)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE CONTRAINT + MERIT TYPE
C ======================================================================
C INP | TYPCON(NUMCON+1) | I*4 | CONSTRAINTS TYPE (1:NUMCON)
C     |                  |     | -> 1=GTE -1=LTE 0=EQU -2=DERIVED DATA
C     |                  |     | MERIT       TYPE (1+NUMCON)
C     |                  |     | -> 1=MAX -1=MIN
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   TYPCON(NUMCON+1)
C ======================================================================
      INTEGER(4)   CON
C ======================================================================
      DO CON = 1, NUMCON + 1
         CONTYP(CON) = TYPCON(CON)
      ENDDO
C ======================================================================
      RETURN
      END
