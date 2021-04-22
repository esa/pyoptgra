      SUBROUTINE OGSENS (CONSTA, CONCON, CONVAR, VARCON, VARVAR)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL SENSITIVITY ANALYSIS
C ======================================================================
C OUT | CONSTA(NUMCON)   | I*4 | CONSTRAINT STATUS (0=PAS 1=ACT)
C OUT | CONCON(NUMCON+1, | R*8 | SENSITIVITY OF CONTRAINTS+MERIT W.R.T.
C     |        NUMCON)   |     |                ACTIVE CONSTRAINTS
C OUT | CONVAR(NUMCON+1, | R*8 | SENSITIVITY OF CONTRAINTS+MERIT W.R.T.
C     |        NUMVAR)   |     |                PARAMETERS
C OUT | VARCON(NUMVAR  , | R*8 | SENSITIVITY OF VARIABLES W.R.T.
C     |        NUMCON)   |     |                ACTIVE CONSTRAINTS
C OUT | VARVAR(NUMVAR  , | R*8 | SENSITIVITY OF VARIABLES W.R.T.
C     |        NUMVAR)   |     |                PARAMETERS
C     |                  |     | -> NOT SCALED
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   CONSTA(NUMCON         )
      REAL(8)      CONCON(NUMCON+1,NUMCON)
      REAL(8)      CONVAR(NUMCON+1,NUMVAR)
      REAL(8)      VARCON(NUMVAR  ,NUMCON)
      REAL(8)      VARVAR(NUMVAR  ,NUMVAR)
C ======================================================================
      REAL(8)       VAL, SCA
      INTEGER(4)    VAR, CON, ACT, PAR, IND, TYP
      CHARACTER     STR*256
C ======================================================================
C CONVERGED
C ----------------------------------------------------------------------
      CONSTA = 0
      DO ACT = 1, NUMACT
         CON = ACTCON(ACT)
         CONSTA(CON) = 1
      ENDDO
C ======================================================================
C SENSITIVITY OF CONTRAINTS W.R.T. ACTIVE CONSTRAINTS
C ----------------------------------------------------------------------
      CONCON = 0D0
      DO CON = 1, NUMCON + 1
         IF (CONACT(CON) .GT. 0) CONCON(CON,CON) = 1D0      
         IF (CONACT(CON) .GT. 0) CYCLE
         CONREF = CONRED(CON,1:NUMACT)
         CALL OGRIGT (CONREF, CONREF)
         DO ACT = 1, NUMACT
            IND = ACTCON(ACT)
            CONCON(CON,IND) = - CONREF(ACT)
         ENDDO
      ENDDO
C ======================================================================
C SENSITIVITY OF CONSTRAINTS W.R.T. PARAMETERS
C ----------------------------------------------------------------------
      CONVAR = 0D0
      DO CON = 1, NUMCON + 1
         IF (CONACT(CON) .GT. 0) CYCLE
         DO VAR = 1, NUMVAR
            IF (VARTYP(VAR) .EQ. 0) CYCLE
            VAL = SENDER(CON,VAR)
            DO ACT = 1, NUMACT
               IND = ACTCON(ACT)
               VAL = VAL + CONCON(CON,IND) * SENDER(IND,VAR)
            ENDDO
            CONVAR(CON,VAR) = VAL
         ENDDO
      ENDDO
C ======================================================================
C SENSITIVITY OF VARIABLES W.R.T. ACTIVE CONSTRAINTS
C ----------------------------------------------------------------------
      VARCON = 0D0
      DO VAR = 1, NUMVAR
         IF (VARTYP(VAR) .NE. 0) CYCLE
         DO ACT = 1, NUMACT
             CON = ACTCON(ACT)
             CONREF(ACT) = CONDER(CON,VAR)
         ENDDO
         CALL OGLEFT (CONREF, CONREF)
         CALL OGRIGT (CONREF, CONREF)
         DO ACT = 1, NUMACT
             CON = ACTCON(ACT)
             VARCON(VAR,CON) = - CONREF(ACT)
         ENDDO
      ENDDO
C ======================================================================
C SENSITIVITY OF VARIABLES W.R.T. PARAMETERS
C ----------------------------------------------------------------------
      VARVAR = 0D0
      DO PAR = 1, NUMVAR
         VARVAR(PAR,PAR) = 1D0     
         IF (VARTYP(PAR) .NE. 1) CYCLE
         DO VAR = 1, NUMVAR
            IF (VARTYP(VAR) .NE. 0) CYCLE
            VAL = 0D0
            DO ACT = 1, NUMACT
               CON = ACTCON(ACT)
               VAL = VAL + VARCON(VAR,CON) * SENDER(CON,PAR)
            ENDDO
            VARVAR(VAR,PAR) = VAL
         ENDDO
      ENDDO
C ======================================================================
C DESCALE SENSITIVITY
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON + 1
         TYP = CONTYP(CON)
         SCA = CONSCA(CON)
         IF (TYP .LT. 0) SCA = - SCA
         CONVAR(CON,1:NUMVAR  ) = CONVAR(CON,1:NUMVAR  ) * SCA
         CONCON(CON,1:NUMCON  ) = CONCON(CON,1:NUMCON  ) * SCA
         IF (CON .GT. NUMCON) CYCLE
         VARCON(1:NUMVAR  ,CON) = VARCON(1:NUMVAR  ,CON) / SCA
         CONCON(1:NUMCON+1,CON) = CONCON(1:NUMCON+1,CON) / SCA
      ENDDO
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         SCA = VARSCA(VAR)
         VARCON(VAR,1:NUMCON  ) = VARCON(VAR,1:NUMCON  ) * SCA
         VARVAR(VAR,1:NUMVAR  ) = VARVAR(VAR,1:NUMVAR  ) * SCA
         CONVAR(1:NUMCON+1,VAR) = CONVAR(1:NUMCON+1,VAR) / SCA
         VARVAR(1:NUMVAR  ,VAR) = VARVAR(1:NUMVAR  ,VAR) / SCA
      ENDDO
C ======================================================================
      RETURN
      END
