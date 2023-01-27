      SUBROUTINE OGEVAL (VALVAR, VALCON, DERVAR, DERCON, CALVAL, CALDER)
C ======================================================================
C COMPUTES SCALED CONTRAINTS+MERIT AND DERIVATIVES
C FROM     SCALED VARIABLES
C ======================================================================
C INP | DERVAR           | I*4 | DERIVATIVES COMPUTATION MODE
C     |                  |     | -> 0: VALUES ONLY
C     |                  |     | -> 1: USER DEFINED
C     |                  |     | -> 2: NUMERIC WITH DOUBLE DIFFERENCING
C     |                  |     | -> 3: NUMERIC WITH SINGLE DIFFERENCING
C ----------------------------------------------------------------------
C INP | CALVAL           | EXT | FUNCTION FOR VALUES
C     |                  |     | -> CALVAL (VALVAR, VALCON)
C     |                  |     | -> INPUT AND OUTPUT NOT SCALED
C ----------------------------------------------------------------------
C INP | CALDER           | EXT | FUNCTION FOR VALUES AND DERIVATIVES
C     |                  |     | -> CALDER (VALVAR, VALCON, DERCON)
C     |                  |     | -> INPUT AND OUTPUT NOT SCALED
C ======================================================================
C SUBROUTINES CALLED: CALVAL, CALDER
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      REAL(8)      VALVAR(NUMVAR)
      REAL(8)      VALCON(NUMCON+1)
      INTEGER(4)   DERVAR
      REAL(8)      DERCON(NUMCON+1,NUMVAR)
      EXTERNAL     CALVAL
      EXTERNAL     CALDER
C ======================================================================
      INTEGER(4)   VAR, CON, COD, LEN, IND, NUMVIO
      REAL(8)      VAL, SCA, FAC, PER, SAV, DER, ERR, CONERR, CONVIO
      CHARACTER    TYP*3, STA*3, NAM*80, STR*256
      REAL(8)      GGG(4,4), BBB(4), VVV(4), OBJVAL
C ======================================================================
      REAL(8),     DIMENSION(:), ALLOCATABLE :: VARVEC
      REAL(8),     DIMENSION(:), ALLOCATABLE :: CONVEC
C ----------------------------------------------------------------------
      ALLOCATE    (VARVEC(NUMVAR))
      ALLOCATE    (CONVEC(NUMCON+1))
C ======================================================================
C GENERAL
C ----------------------------------------------------------------------
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      IF     (DERVAR .EQ. 0) THEN
         WRITE (STR,'("COMPUTE RESULTS")')
         CALL OGWRIT (3,STR)
      ELSEIF (DERVAR .EQ. 1 .OR. DERVAR .EQ. -1) THEN
         WRITE (STR,'("COMPUTE RESULTS",'//
     &              '   " AND DERIVATIVES USER DEFINED")')
         CALL OGWRIT (3,STR)
      ELSEIF (DERVAR .EQ. 2) THEN
         WRITE (STR,'("COMPUTE RESULTS",'//
     &              '   " AND DERIVATIVES BY DOUBLE DIFFERENCING")')
         CALL OGWRIT (3,STR)
      ELSEIF (DERVAR .EQ. 3) THEN
         WRITE (STR,'("COMPUTE RESULTS",'//
     &              '   " AND DERIVATIVES BY SINGLE DIFFERENCING")')
         CALL OGWRIT (3,STR)
      ENDIF
C ======================================================================
C WRITE VARIABLES
C ----------------------------------------------------------------------
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      WRITE (STR,'("VARIABLES NOT SCALED:")')
      CALL OGWRIT (3,STR)
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      DO VAR = 1, NUMVAR
         VAL = VALVAR(VAR)
         SCA = VARSCA(VAR)
         COD = VARTYP(VAR)
         IF (COD .EQ. 0) TYP = "FRE"
         IF (COD .EQ. 1) TYP = "PAR"
         NAM = VARSTR(VAR)
         LEN = VARLEN(VAR)
         WRITE (STR,
     &        '("VAR/VAL/SCA/TYP/NAM=",'//
     &        '  I5,D14.6,D9.1,1X,A3,1X,A)')
     &           VAR,VAL*SCA,SCA,TYP,NAM(1:LEN)
         CALL OGWRIT (3,STR)
      ENDDO
C ======================================================================
C DE-SCALE VARIABLES
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         VARVEC(VAR) = VALVAR(VAR) * VARSCA(VAR)
      ENDDO
C ======================================================================
C GET RESULTS
C GET DERIVATIVES IF USER DEFINED
C ----------------------------------------------------------------------
      IF     (DERVAR .EQ. 0) THEN
         CALL CALVAL (VARVEC, VALCON, 0)
      ELSEIF (DERVAR .EQ. 1 .OR. DERVAR .EQ. -1) THEN
         CALL CALVAL (VARVEC, VALCON, 1)
      ELSEIF (DERVAR .EQ. 2) THEN
         CALL CALVAL (VARVEC, VALCON, 1)
      ELSEIF (DERVAR .EQ. 3) THEN
         CALL CALVAL (VARVEC, VALCON, 1)
      ENDIF
C ======================================================================
      IF (1 .EQ. 2) THEN
      GGG(1,1) = +1D+01
      GGG(2,1) = +1D+00
      GGG(3,1) = +2D+00
      GGG(4,1) = +3D+00
      GGG(1,2) = GGG(2,1)
      GGG(2,2) = +1D+01
      GGG(3,2) = +4D+00
      GGG(4,2) = +5D+00
      GGG(1,3) = GGG(3,1)
      GGG(2,3) = GGG(3,2)
      GGG(3,3) = +1D+01
      GGG(4,3) = +6D+00
      GGG(1,4) = GGG(4,1)
      GGG(2,4) = GGG(4,2)
      GGG(3,4) = GGG(4,3)
      GGG(4,4) = +1D+01
C ----------------------------------------------------------------------
      BBB(1) = +1D+01
      BBB(2) = +1D+01
      BBB(3) = +1D+01
      BBB(4) = +1D+01
C ----------------------------------------------------------------------
      CALL MUL2M (GGG   , 4, 1, 1, 4,
     &            VALVAR, 4, 1, 1, 4,
     &            VVV   , 4, 1, 1, 1)
      CALL MUL2M (VALVAR, 1, 1, 1, 1,
     &            VVV   , 4, 1, 1, 4,
     &            VALCON, 1, 1, 1, 1)
      CALL MUL2M (BBB   , 1, 1, 1, 1,
     &            VALVAR, 4, 1, 1, 4,
     &            VVV   , 1, 1, 1, 1)
      CALL MULVS (VALCON, 0.5D0, VALCON, 1)
      CALL SUM2V (VALCON, VVV, VALCON, 1)
      CALL MULVS (VALCON, -1D0, VALCON, 1)
      WRITE (STR,*) "VALCON=",(VALCON(IND),IND=1,1)
      CALL OGWRIT (3,STR)
      ENDIF
C ----------------------------------------------------------------------
C SCALE RESULTS
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON + 1
         CONVEC(CON) = VALCON(CON)
         SCA = CONSCA(CON)
         COD = CONTYP(CON)
         IF (COD .EQ. -1) SCA = - SCA
         VALCON(CON) = VALCON(CON) / SCA
      ENDDO
C ======================================================================
C WRITE RESULTS
C ----------------------------------------------------------------------
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      WRITE (STR,'("RESULTS NOT SCALED:")')
      CALL OGWRIT (3,STR)
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      CONERR = 0D0  ! total constraint error (scaled to constr. threshod)
      CONVIO = 0D0  ! total constaint error norm (unscaled)
      IND = 0  ! index of largest constraint violation
      FAC = 0D0  ! value of largest constraint violation
      NUMVIO = 0  ! number of violated constraints
      DO CON = 1, NUMCON + 1
         VAL = VALCON(CON)
         SCA = CONSCA(CON)
         COD = CONTYP(CON)
         STA = "   "
         ERR = 0D0
         IF (COD .EQ. -1) SCA = - SCA
         IF (COD .EQ. -2                      ) TYP = "DER"
         IF (COD .EQ. -1 .AND. CON .LE. NUMCON) TYP = "LTE"
         IF (COD .EQ.  0 .AND. CON .LE. NUMCON) TYP = "EQU"
         IF (COD .EQ.  1 .AND. CON .LE. NUMCON) TYP = "GTE"
         IF (COD .EQ. -1 .AND. CON .GT. NUMCON) TYP = "MIN"
         IF (COD .EQ.  1 .AND. CON .GT. NUMCON) TYP = "MAX"
         IF (COD .EQ.  0 .AND. CON .LE. NUMCON .AND.
     &                      DABS(VAL) .GT. 1D0) THEN
            STA = "VIO"
            ERR = DABS (VAL)
            NUMVIO = NUMVIO+1
         ENDIF
         IF (COD .NE.  0 .AND. CON .LE. NUMCON .AND.
     &       COD .NE. -2 .AND. -VAL  .GT. 1D0) THEN
            STA = "VIO"
            ERR = DABS (VAL)
            NUMVIO = NUMVIO+1
         ENDIF
         CONERR = CONERR + ERR
         CONVIO = CONVIO + (ERR * SCA)**2
         IF (ERR .GT. FAC) IND = CON
         IF (ERR .GT. FAC) FAC = ERR
         NAM = CONSTR(CON)
         LEN = CONLEN(CON)
         WRITE (STR,
     &        '("CON/VAL/SCA/TYP/STA/NAM=",'//
     &        '  I5,D14.6,D9.1,1X,A3,1X,A3,1X,A)')
     &           CON,VAL*SCA,SCA,TYP,STA,NAM(1:LEN)
         CALL OGWRIT (3,STR)
      ENDDO
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      WRITE (STR,'("CONSTRAINT ERROR.:",2(1X,D13.6),I6)')
     &                 CONERR,FAC,IND
      CALL OGWRIT (3,STR)
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
C write pygmo-style log output
      OBJVAL = -VALCON(NUMCON+1)
      CONVIO = DSQRT(CONVIO)
      CALL OGPWRI(OBJVAL, NUMVIO, CONVIO, DERVAR)
C ======================================================================
C NO DERIVATIVES
C ----------------------------------------------------------------------
      IF     (DERVAR .EQ. 0) THEN
         RETURN
      ELSEIF (DERVAR .EQ. 1 .OR. DERVAR .EQ. -1) THEN
         CALL CALDER (VARVEC, CONVEC, DERCON)
      ENDIF
C ----------------------------------------------------------------------
      IF (1 .EQ. 2) THEN
      CALL MUL2M (VALVAR, 1, 1, 1, 1,
     &            GGG   , 4, 1, 1, 4,
     &            DERCON, 1, 1, 1, 4)
      CALL SUM2V (DERCON, BBB, DERCON, 4)
      CALL MULVS (DERCON, -1D0, DERCON, 4)
      WRITE (STR,*) "DERCON=",(DERCON(1,IND),IND=1,4)
      CALL OGWRIT (3,STR)
      ENDIF
C ======================================================================
C WRITE DERIVATIVES
C ----------------------------------------------------------------------
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
      WRITE (STR,'("DERIVATIVES SCALED:")')
      CALL OGWRIT (3,STR)
      WRITE (STR,'()')
      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
C ----------------------------------------------------------------------
C WRITE VARIABLE
C ----------------------------------------------------------------------
         VAL = VALVAR(VAR)
         SCA = VARSCA(VAR)
         COD = VARTYP(VAR)
         IF (COD .EQ. 0) TYP = "FRE"
         IF (COD .EQ. 1) TYP = "PAR"
         NAM = VARSTR(VAR)
         LEN = VARLEN(VAR)
         WRITE (STR,
     &        '("VAR/VAL/SCA/TYP/NAM=",'//
     &        '  I5,D14.6,D9.1,1X,A3,1X,A)')
     &           VAR,VAL*SCA,SCA,TYP,NAM(1:LEN)
         CALL OGWRIT (4,STR)
         WRITE (STR,'()')
         CALL OGWRIT (4,STR)
C ----------------------------------------------------------------------
C DERIVATIVES BY DOUBLE DIFFERENCING
C ----------------------------------------------------------------------
         IF (DERVAR .EQ. 2) THEN
C ----------------------------------------------------------------------
            PER = VARPER(VAR)
            SAV = VARVEC(VAR)
            VARVEC(VAR) = SAV + PER
            CALL CALVAL (VARVEC, DERCON(1,VAR), 0)
            VARVEC(VAR) = SAV - PER
            CALL CALVAL (VARVEC, CONVEC, 0)
            FAC = 0.5D0 / PER
            DO CON = 1, NUMCON + 1
               DERCON(CON,VAR) = (DERCON(CON,VAR) - CONVEC(CON)) * FAC
            ENDDO
            VARVEC(VAR) = SAV
C ----------------------------------------------------------------------
         ENDIF
C ----------------------------------------------------------------------
C DERIVATIVES BY SINGLE DIFFERENCING
C ----------------------------------------------------------------------
         IF (DERVAR .EQ. 3) THEN
C ----------------------------------------------------------------------
            PER = VARPER(VAR)
            SAV = VARVEC(VAR)
            VARVEC(VAR) = SAV + PER
            CALL CALVAL (VARVEC, DERCON(1,VAR), 0)
            FAC = 1.0D0 / PER
            DO CON = 1, NUMCON + 1
               DERCON(CON,VAR) = (DERCON(CON,VAR) - CONVEC(CON)) * FAC
            ENDDO
            VARVEC(VAR) = SAV
C ----------------------------------------------------------------------
         ENDIF
C ----------------------------------------------------------------------
C SCALE DERIVATIVES
C ----------------------------------------------------------------------
         DO CON = 1, NUMCON + 1
            FAC = VARSCA(VAR) / CONSCA(CON)
            IF (CONTYP(CON) .EQ. -1) FAC = - FAC
            DERCON(CON,VAR) = DERCON(CON,VAR) * FAC
         ENDDO
C ----------------------------------------------------------------------
C WRITE DERIVATIVES
C ======================================================================
         DO CON = 1, NUMCON + 1
            DER = DERCON(CON,VAR)
            IF (DER .NE. 0D0) THEN
               SCA = CONSCA(CON)
               COD = CONTYP(CON)
               IF (COD .EQ. -1) SCA = - SCA
               IF (COD .EQ. -2                      ) TYP = "DER"
               IF (COD .EQ. -1 .AND. CON .LE. NUMCON) TYP = "LTE"
               IF (COD .EQ.  0 .AND. CON .LE. NUMCON) TYP = "EQU"
               IF (COD .EQ.  1 .AND. CON .LE. NUMCON) TYP = "GTE"
               IF (COD .EQ. -1 .AND. CON .GT. NUMCON) TYP = "MIN"
               IF (COD .EQ.  1 .AND. CON .GT. NUMCON) TYP = "MAX"
               NAM = CONSTR(CON)
               LEN = CONLEN(CON)
               WRITE (STR,
     &              '("CON/DER/SCA/TYP/NAM=",'//
     &              '  I5,D14.6,D9.1,1X,A3,1X,A)')
     &                 CON,DER*SCA/VARSCA(VAR),SCA,TYP,NAM(1:LEN)
               CALL OGWRIT (4,STR)
            ENDIF
         ENDDO
         WRITE (STR,'()')
         CALL OGWRIT (4,STR)
C ----------------------------------------------------------------------
      ENDDO
C ======================================================================
      DEALLOCATE  (VARVEC)
      DEALLOCATE  (CONVEC)
C ======================================================================
      RETURN
      END
