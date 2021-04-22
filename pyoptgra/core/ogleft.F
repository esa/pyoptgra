      SUBROUTINE OGLEFT (ACTINP, ACTOUT)
C ======================================================================
C LEFT-MULTIPLIES VECTOR LOWER TRIANGULAR MATRIX OBTAINED BY REDUCTION
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
C ======================================================================
      DO ACT = 1, NUMACT
         ROW = ACTCON(ACT)
         VAL = ACTINP(ACT)
         DO COL = 1, ACT - 1
            VAL = VAL - CONRED(ROW,COL) * ACTOUT(COL)
         ENDDO
         ACTOUT(ACT) = VAL / CONRED(ROW,ACT)
      ENDDO
C ======================================================================
      RETURN
      END
