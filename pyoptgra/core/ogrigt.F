      SUBROUTINE OGRIGT (ACTINP, ACTOUT)
C ======================================================================
C RIGHT-MULTIPLIES VECTOR LOWER TRIANGULAR MATRIX OBTAINED BY REDUCTION
C AND SUBSEQUENT INVERSION OF DERIVATIVES OF ACTIVE CONSTRAINTS
C ======================================================================
C INP | ACTINP(NUMCON)   | R*8 | VECTOR INITAL
C ----------------------------------------------------------------------
C OUT | ACTOUT(NUMCON)   | R*8 | VECTOR FINAL (MAY BE SAME AS ACTINP)
C ======================================================================
C SUBROUTINES CALLED: NONE
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      REAL(8)      ACTINP(NUMCON)
      REAL(8)      ACTOUT(NUMCON)
C ======================================================================
      INTEGER(4)   ROW, COL, ACT
      REAL(8)      VAL
C =====================================================================
      DO COL = NUMACT, 1, -1
         VAL = ACTINP(COL)
         DO ACT = NUMACT, COL + 1, -1
            ROW = ACTCON(ACT)
            VAL = VAL - CONRED(ROW,COL) * ACTOUT(ACT)
         ENDDO
         ROW = ACTCON(COL)
         ACTOUT(COL) = VAL / CONRED(ROW,COL)
      ENDDO
C ======================================================================
      RETURN
      END
