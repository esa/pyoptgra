      SUBROUTINE OGINCL (INC)
C ======================================================================
C ADDS CONSTRAINT TO ACTIVE SET AND REDUCES DERIVATIVES
C ======================================================================
C INP | INC              | I*4 | CONSTRAINT TO BE INCLUDED
C ======================================================================
C SUBROUTINES CALLED: NONE
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4) INC
C ======================================================================
      REAL(8)    VAL, FAC, GAM, SAV, MAX
      INTEGER(4) ROW, COL, IND, LST
      CHARACTER  STR*256
C ======================================================================
C GENERAL
C ----------------------------------------------------------------------
      NUMACT = NUMACT + 1
C ======================================================================
C PERMUTATION TO GET ZERO DERIVATIVES AT END FOR NEW ACTIVE CONSTRAINT
C ----------------------------------------------------------------------
      LST = NUMVAR
      DO COL = NUMVAR, NUMACT, -1
         IF (CONRED(INC,COL) .EQ. 0D0) THEN
            IF (COL .NE. LST) THEN
               DO ROW = 1, NUMCON + 3
                  IF (CONACT(ROW) .LE. 0) THEN
                     SAV = CONRED(ROW,COL)
                     CONRED(ROW,COL) = CONRED(ROW,LST)
                     CONRED(ROW,LST) = SAV
                  ENDIF
               ENDDO
            ENDIF
            LST = LST - 1
         ENDIF
      ENDDO
C ======================================================================
C PERMUTATION TO GET MAXIMUM PIVOT
C ----------------------------------------------------------------------
      IND = NUMACT
      MAX = DABS (CONRED(INC,IND))
      DO COL = NUMACT + 1, LST
         VAL = DABS (CONRED(INC,COL))
         IF (VAL .GT. MAX) THEN
            IND = COL
            MAX = VAL
         ENDIF
      ENDDO
C ----------------------------------------------------------------------
      IF (IND .NE. NUMACT) THEN
         DO ROW = 1, NUMCON + 3
            IF (CONACT(ROW) .LE. 0) THEN
               SAV = CONRED(ROW,IND)
               CONRED(ROW,IND) = CONRED(ROW,NUMACT)
               CONRED(ROW,NUMACT) = SAV
            ENDIF
         ENDDO
      ENDIF
C ======================================================================
C UPDATE LIST OF ACTIVE CONSTRAINTS
C ----------------------------------------------------------------------
      ACTCON(NUMACT) = INC
      CONACT(INC) = NUMACT
C ======================================================================
C REDUCE FOR NEW ACTIVE CONSTRAINT
C ----------------------------------------------------------------------
      IF (DABS (CONRED(INC,NUMACT)) .LT. 1D-12) THEN
         WRITE (STR,*) "OGINCL-WARNING: CONSTRAINT SINGULAR"
         CALL OGWRIT (2,STR)
         WRITE (STR,*) "INC=",INC
         CALL OGWRIT (2,STR)
         WRITE (STR,*) "PIV=",CONRED(INC,NUMACT)
         CALL OGWRIT (2,STR)
         NUMACT = NUMACT - 1
         CONACT(INC) = 0
         RETURN
      ENDIF
C ----------------------------------------------------------------------
      VAL = DSQRT(SUM(CONRED(INC,NUMACT:LST)**2))
      IF (CONRED(INC,NUMACT) .GT. 0D0) VAL = - VAL
C ----------------------------------------------------------------------
      CONRED(INC,NUMACT) = CONRED(INC,NUMACT) - VAL
C ----------------------------------------------------------------------
      SAV = CONRED(INC,NUMACT)
      FAC = 1D0 / SAV
      CONRED(INC,NUMACT:LST) = CONRED(INC,NUMACT:LST) * FAC
C ----------------------------------------------------------------------
      FAC = SAV / VAL
      DO ROW = 1, NUMCON + 3
         IF (CONACT(ROW) .LE. 0) THEN
            GAM = DOT_PRODUCT (CONRED(ROW,NUMACT:LST),
     &                         CONRED(INC,NUMACT:LST))
            IF (GAM .NE. 0D0) THEN
               GAM = GAM * FAC
               CONRED(ROW,NUMACT:LST) = CONRED(ROW,NUMACT:LST)
     &                                + CONRED(INC,NUMACT:LST) * GAM
            ENDIF
         ENDIF
      ENDDO
C ----------------------------------------------------------------------
      CONRED(INC,NUMACT) = VAL
      CONRED(INC,NUMACT+1:LST) = 0D0
C ======================================================================
      RETURN
      END
