      SUBROUTINE OGVSTR (STRVAR, LENVAR)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE VARIABLE STRING
C ======================================================================
C INP | STRVAR(NUMVAR)   | C80 | VARIABLES NAME STRING
C ----------------------------------------------------------------------
C INP | LENVAR(NUMVAR)   | I*4 | VARIABLES NAME LENGTH
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      CHARACTER*80 STRVAR(NUMVAR)
      INTEGER(4)   LENVAR(NUMVAR)
C ======================================================================
      INTEGER(4)   VAR, LEN
C ======================================================================
      DO VAR = 1, NUMVAR
         LEN = MIN (LENVAR(VAR), 80)
         VARSTR(VAR) = STRVAR(VAR)
         VARLEN(VAR) = LEN
      ENDDO
C ======================================================================
      RETURN
      END
