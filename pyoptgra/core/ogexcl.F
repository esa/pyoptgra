      SUBROUTINE OGEXCL (EXC)
C ======================================================================
C REMOVE CONSTRAINT TO ACTIVE SET AND REDUCES DERIVATIVES
C ======================================================================
C INP | EXC              | I*4 | CONSTRAINT TO BE REMOVED
C     |                  |     | SEQUENCE NUMBER IN ACTIVE LIST
C ======================================================================
C SUBROUTINES CALLED: NONE
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4) EXC
C ======================================================================
      REAL(8)    VAL, BET, GAM
      INTEGER(4) ROW, COL, IND, ACT, CON
      CHARACTER  STR*256
C ======================================================================
C ADJUST LIST OF ACTIVE CONSTRAINTS
C ----------------------------------------------------------------------
      CON = ACTCON(EXC)
      CONACT(CON) = 0
      NUMACT = NUMACT - 1
      DO ACT = EXC, NUMACT
         CON = ACTCON(ACT+1)
         ACTCON(ACT) = CON
         CONACT(CON) = CONACT(CON) - 1
      ENDDO
C ======================================================================
C REDUCE FOR SUBSEQUENT CONSTRAINTS
C ----------------------------------------------------------------------
      DO ACT = EXC, NUMACT
         CON = ACTCON(ACT)
         VAL = 0D0
         DO COL = ACT, ACT + 1
            VAL = VAL + CONRED(CON,COL) ** 2
         ENDDO
         VAL = DSQRT (VAL)
         IF (CONRED(CON,ACT) .GT. 0D0) VAL = - VAL
         IF (DABS (VAL) .LT. 1D-15) THEN
            WRITE (LOGLUN,*) "OGEXCL-ERROR: CONSTRAINTS SINGULAR"
            CALL OGWRIT (2,STR)
            WRITE (LOGLUN,*) "VAL=",VAL
            CALL OGWRIT (2,STR)
            STOP
         ENDIF
         CONRED(CON,ACT) = CONRED(CON,ACT) - VAL
         BET = 1D0 / (VAL * CONRED(CON,ACT))
         DO ROW = 1, NUMCON + 3
            IF (CONACT(ROW) .GT. ACT .OR. CONACT(ROW) .LE. 0) THEN
               GAM = 0D0
               DO COL = ACT, ACT + 1
                  IF (CONRED(ROW,COL) .NE. 0D0) THEN
                     GAM = GAM + CONRED(ROW,COL) * CONRED(CON,COL)
                  ENDIF
               ENDDO
               IF (GAM .NE. 0D0) THEN
                  GAM = GAM * BET
                  DO COL = ACT, ACT + 1
                     CONRED(ROW,COL) = CONRED(ROW,COL)
     &                               + CONRED(CON,COL) * GAM
                  ENDDO
               ENDIF
            ENDIF
         ENDDO
         CONRED(CON,ACT) = VAL
         DO COL = ACT + 1, ACT + 1
            CONRED(CON,COL) = 0D0
         ENDDO
      ENDDO
C ======================================================================
      RETURN
      END
