      SUBROUTINE OGSENS (CONSEN, ACTNUM, ACTLST)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN
C ======================================================================
C OUT | CONSEN(NUMCON+1, | R*8 | SENSITIVITY:
C     |        NUMVAR)   |     | -> FOR CONSTRAINTS (1:NUMCON,1:NUMVAR)
C     |                  |     | -> FOR MERIT       (1+NUMCON,1:NUMVAR)
C     |                  |     | -> SENSITIVITY TO ACTIVE CONSTRAINT "I"
C     |                  |     |    IS IN COLUMN OF FREE VARIABLE "I"
C     |                  |     |    (NOT OF VARIABLE "I" !!!)
C     |                  |     | -> SENSITIVITY TO PARAMETER VARIABLE
C     |                  |     |    IS IN COLUMN OF PARAMETER VARIABLE
C     |                  |     | -> NOT SCALED
C ----------------------------------------------------------------------
C OUT | ACTNUM           | I*4 | NUMBER OF ACTIVE CONSTRAINTS
C ----------------------------------------------------------------------
C OUT | ACTLST(NUMVAR)   | I*4 | LIST OF ACTIVE CONSTRAINTS
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
      CHARACTER*160 SCCS_INFO(2)
      DATA SCCS_INFO
     &/'%Z%%P% ',
     &'%I%      %E%     MAN_FDD\n'/
C ======================================================================
#include 'ogdata.inc'
C ======================================================================
      INTEGER*4    CONSEN(NUMCON+1,NUMVAR)
      INTEGER*4    ACTNUM
      INTEGER*4    ACTLST(NUMVAR)
C ======================================================================
      INTEGER*4    VAR, CON, ACT, ROW, TYP, INC, LEN
      REAL*8       VAL, SCA
C ======================================================================
C CONVERGED
C ----------------------------------------------------------------------
      IF (CONVER .EQ. 0) THEN
         WRITE (LOGLUN,*) ""
         WRITE (LOGLUN,*) "OGSENS-ERROR: SOLUTION NOT CONVERGED"
         CALL flush (LOGLUN)
         RETURN
      ENDIF
C ======================================================================
C SENSITIVITY
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON + 1
         IF (CONACT(CON) .EQ. 0) THEN
            VAR = 1
            DO ACT = 1, NUMACT
               DO WHILE (VARTYP(VAR) .NE. 0)
                  VAR = VAR + 1
               ENDDO
               CONREF(ACT) = CONRED(CON,VAR)
               VAR = VAR + 1
            ENDDO
            CALL OGRIGT (CONREF, CONREF)
            VAR = 1
            DO ACT = 1, NUMACT
               DO WHILE (VARTYP(VAR) .NE. 0)
                  VAR = VAR + 1
               ENDDO
               CONSEN(CON,VAR) = - CONREF(ACT)
               VAR = VAR + 1
            ENDDO
            DO VAR = 1, NUMVAR
               VAL = CONDER(CON,VAR)
               IF (VARTYP(VAR) .EQ. 1) THEN
                  DO ACT = 1, NUMACT
                     ROW = ACTCON(ACT)
                     VAL = VAL + CONDER(ROW,VAR) * CONREF(ACT)
                  ENDDO
                  CONSEN(CON,VAR) = VAL
               ENDIF
            ENDDO
         ENDIF
      ENDDO
C ----------------------------------------------------------------------
      DO ACT = 1, NUMACT
         CON = ACTCON(ACT)
         DO VAR = 1, NUMVAR
            CONSEN(CON,VAR) = 0D0
         ENDDO
      ENDDO
C ----------------------------------------------------------------------
      VAR = 1
      DO ACT = 1, NUMACT
         DO WHILE (VARTYP(VAR) .NE. 0)
            VAR = VAR + 1
         ENDDO
         CON = ACTCON(ACT)
         CONSEN(CON,VAR) = 1D0
         VAR = VAR + 1
      ENDDO
C ======================================================================
C DESCALE SENSITIVITY
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON + 1
         TYP = CONTYP(CON)
         SCA = CONSCA(CON)
         IF (TYP .LT. 0) SCA = - SCA
         DO VAR = 1, NUMVAR
            CONSEN(CON,VAR) = CONSEN(CON,VAR) * SCA
         ENDDO
      ENDDO
C ----------------------------------------------------------------------
      VAR = 1
      DO ACT = 1, NUMACT
         DO WHILE (VARTYP(VAR) .NE. 0)
            VAR = VAR + 1
         ENDDO
         CON = ACTCON(ACT)
         TYP = CONTYP(CON)
         SCA = CONSCA(CON)
         IF (TYP .LT. 0) SCA = - SCA
         DO ROW = 1, NUMCON + 1
            CONSEN(ROW,VAR) = CONSEN(ROW,VAR) / SCA
         ENDDO
         VAR = VAR + 1
      ENDDO
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         IF (VARTYP(VAR) .EQ. 1) THEN
            SCA = VARSCA(VAR)
            DO ROW = 1, NUMCON + 1
               CONSEN(ROW,VAR) = CONSEN(ROW,VAR) / SCA
            ENDDO
         ENDIF
      ENDDO
C ======================================================================
      ACTNUM = NUMACT
      DO ACT = 1, NUMACT
         ACTLST(ACT) = ACTCON(ACT)
      ENDDO
C ======================================================================
      RETURN
      END
