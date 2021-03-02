      SUBROUTINE OGEXEC (VALVAR, VALCON, FINOPT, FINITE, CALVAL, CALDER)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN
C ======================================================================
C I/O | VALVAR(NUMVAR)   | R*8 | VARIABLES VALUE
C     |                  |     | -> NOT SCALED
C ----------------------------------------------------------------------
C OUT | VALCON(NUMCON+1) | R*8 | CONSTRAINTS VALUE (1:NUMCON)
C     |                  |     | MERIT       VALUE (1+NUMCON)
C     |                  |     | -> NOT SCALED
C ----------------------------------------------------------------------
C OUT | FINOPT           | I*4 | TERMINATION STATUS
C     |                  |     | -> 1=    MATCHED &     OPTIMAL
C     |                  |     | -> 2=    MATCHED & NOT OPTIMAL
C     |                  |     | -> 3=NOT MATCHED & NOT OPTIMAL
C     |                  |     | -> 4=NOT FEASIBL & NOT OPTIMAL
C ----------------------------------------------------------------------
C INP | CALVAL           | EXT | FUNCTION FOR VALUES
C     |                  |     | -> CALDER (VALVAR, VALCON)
C     |                  |     | -> INPUT AND OUTPUT NOT SCALED
C ----------------------------------------------------------------------
C INP | CALDER           | EXT | FUNCTION FOR VALUES AND DERIVATIVES
C     |                  |     | -> CALDER (VALVAR, VALCON, CONDER)
C     |                  |     | -> INPUT AND OUTPUT NOT SCALED
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
      REAL*8       VALVAR(NUMVAR)
      REAL*8       VALCON(NUMCON+1)
      INTEGER*4    FINOPT, FINISH, FINITE, ITECOR, ITEOPT
      EXTERNAL     CALVAL
      EXTERNAL     CALDER
C ======================================================================
      INTEGER*4    ITE, VAR, CON, ACT, ROW, TYP, INC, LEN, NUM
      REAL*8       VAL, SCA, RED, DER, FAC, OLD
      CHARACTER*256 NAM,STR
C ----------------------------------------------------------------------
      INTEGER*4    NUMINC, NUMEQU, NUMITE, ITEDIV, ITECNV
      REAL*8       VARACC, VARDIS, COSNEW, COSOLD, VARSAV, MEAMER
      REAL*8       CONERR, DESNOR, NORERR, MEAERR
C ======================================================================
      REAL*8,      DIMENSION(:  ), ALLOCATABLE :: VARSUM
      REAL*8,      DIMENSION(:  ), ALLOCATABLE :: VARCOR
      REAL*8,      DIMENSION(:  ), ALLOCATABLE :: CONCOR
C ----------------------------------------------------------------------
      ALLOCATE    (VARSUM(NUMVAR))
      ALLOCATE    (VARCOR(NUMVAR))
      ALLOCATE    (CONCOR(NUMCON+1))
C ======================================================================
C GENERAL
C ----------------------------------------------------------------------
C     LOGLEV = 2
      CALL OGWRIT (2,"")
      CALL OGWRIT (2,"OPTGRA START")
      CALL OGWRIT (2,"")
      CALL flush  (LOGLUN)
      FINOPT = 3
      ITECOR = 0
      ITEOPT = 0
      MEAERR = 0D0
      MEAMER = 0D0
      ITEDIV = 0
      ITECNV = 0
      CONOPT = 0
      CONCOR = 0D0
      VARCOR = 0D0
C ----------------------------------------------------------------------
      VARSTP = VARSND
      NUMITE = 0
      COSNEW = 0D0
      DO VAR = 1, NUMVAR
          VARSUM(VAR) = 0D0
          VARCOR(VAR) = 0D0
      ENDDO
C ======================================================================
C EQUALTIY CONSTRAINTS IN ACTIVE SET
C ----------------------------------------------------------------------
      NUMACT = 0
      DO CON = 1, NUMCON   
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,*) "CON/PRI=",CON,CONPRI(CON)," ", NAM(1:LEN)
          CALL OGWRIT (3,STR)
          CONACT(CON) = 0
          IF (CONSCA(CON) .GE. 1D+09) CONTYP(CON) = -2
          IF     (CONTYP(CON) .EQ.  0) THEN
          ELSEIF (CONTYP(CON) .EQ. -2) THEN
              CONACT(CON) = -2
          ELSE
          ENDIF
      ENDDO
      NUMEQU = NUMACT
      CONACT(NUMCON+1) = -3
      CONACT(NUMCON+2) = -3
C ======================================================================
C SCALE VARIABLES
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
          VARVAL(VAR) = VALVAR(VAR) / VARSCA(VAR)
      ENDDO
C ======================================================================
C HEADER FOR TABLE
C ----------------------------------------------------------------------
      IF (TABLEV .GE. 1) THEN
          WRITE (TABLUN,'("ITER",1X,"OPT",1X,1000(1X,I10))')
     &    (VAR,VAR=1,NUMVAR),
     &    (CON,CON=1,NUMCON)
      ENDIF
C ======================================================================
 1000 CONTINUE
C      IF (NUMITE .GE. 89) MATLEV = 3
C ======================================================================
C NEW ITERATION
C ----------------------------------------------------------------------
      IF     (NUMITE .GE. CORITE .AND. ITECOR .EQ. 0) THEN
          FINOPT = 3
          FINITE = NUMITE
          CALL OGWRIT (1,"")
          WRITE(STR,'("OPTGRA: Converged: not ITERAT=",2I4,2D11.3)')
     &    NUMITE,MAXITE,CONERR,DESNOR
          CALL OGWRIT (1,STR)
          GOTO 9999
      ELSEIF (NUMITE .GE. MAXITE .OR.
     &       (NUMITE-ITECOR .GE. OPTITE-1 .AND. ITECOR .NE. 0)) THEN
          FINOPT = 2
          FINITE = ITEOPT
          VARVAL = VARCOR
          CONVAL = CONCOR
          CALL OGWRIT (1,"")
          WRITE(STR,'("OPTGRA: Converged: mat ITERAT=",2I4,2D11.3)')
     &    NUMITE,MAXITE,CONERR,DESNOR
          CALL OGWRIT (1,STR)
          GOTO 9999
      ENDIF
C ----------------------------------------------------------------------
      NUMITE = NUMITE + 1
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'("ITERAT=",I5)') NUMITE
      CALL OGWRIT (2,STR)
C ======================================================================
C GET VALUES AND GRADIENTS
C ======================================================================
C      IF (NUMITE .EQ. 1) MATLEV = 3
C      IF (NUMITE .GT. 1) MATLEV = 2
      CALL OGEVAL (VARVAL, CONVAL, VARDER, CONDER(1:NUMCON+1,:),
     &             CALVAL, CALDER)
C      MATLEV = 2
C ======================================================================
      IF (VARDER .EQ. -1) THEN
          CONRED(1:NUMCON+1,:) = CONDER(1:NUMCON+1,:)
          CALL OGEVAL (VARVAL, CONVAL, 2, CONDER(1:NUMCON+1,:),
     &                 CALVAL, CALDER)
          WRITE (STR,'("GRADIENT CHECK")')
          CALL OGWRIT (1,STR)
          DO VAR = 1,NUMVAR
				DO CON = 1,NUMCON+1
				FAC = VARSCA(VAR) / CONSCA(CON)
				FAC = 1D0
                  DER = CONDER(CON,VAR) * FAC 
                  RED = CONRED(CON,VAR) * FAC
                  IF (ABS(DER) .LT. 1D-6 .AND. ABS(RED) .LT. 1D-6) CYCLE
                  IF (DABS(DER-RED) .LT. 1D-2) CYCLE
                  IF (DER .NE. 0D0) THEN
                      FAC = RED/DER
                  ELSE
                      FAC = 0D0
                  ENDIF
                  IF (DABS(FAC-1D0) .LT. 1D-2) CYCLE
				WRITE (STR,'("VAR/CON/ANA/NUM/A2N=",2I4,3(1X,D13.6))')
     &            VAR,CON,RED,DER,FAC
                  CALL OGWRIT (1,STR)
                  NAM = VARSTR(VAR)
                  LEN = VARLEN(VAR)
                  WRITE (STR,'("      VAR=",A,1X,D13.6)') NAM(1:LEN),
     &            VARVAL(VAR)*VARSCA(VAR)
                  CALL OGWRIT (1,STR)
                  NAM = CONSTR(CON)
                  LEN = CONLEN(CON)
                  WRITE (STR,'("      CON=",A,1X,D13.6)') NAM(1:LEN),
     &            CONVAL(CON)*CONSCA(CON)
                  CALL OGWRIT (1,STR)
              ENDDO
          ENDDO
C          CONDER(1:NUMCON+1,:) = CONRED(1:NUMCON+1,:)
C		GOTO 9999
      ENDIF
C ======================================================================
      IF (NUMITE .EQ. 1) THEN
          VARGRD = VARVAL
      ELSE
          VARGRD = VARREF
      ENDIF
      VARREF = VARVAL
      CONREF = CONVAL
C ======================================================================
      VARACC = 0D0
C ======================================================================
      COSOLD = COSNEW
      COSNEW = CONVAL(NUMCON+1)
      WRITE (STR,'("OPTGRA: VALCOS=",D15.8,1X,D15.8)') COSNEW,
     &COSNEW-COSOLD
      CALL OGWRIT (3,STR)
C ======================================================================
C CORRECTION PART
C ----------------------------------------------------------------------
      CALL OGCORR (VARACC, FINISH, CONERR,NORERR,CALVAL,CALDER)
C ----------------------------------------------------------------------
      IF (TABLEV .GE. 1) THEN
          WRITE (TABLUN,'(I4,1X,"COR",1X,1000(1X,D10.3))') NUMITE,
     &    (VARVAL(VAR),VAR=1,NUMVAR),
     &    (CONVAL(CON),CON=1,NUMCON)
      ENDIF
C ----------------------------------------------------------------------
      IF (FINISH .EQ. 0) THEN
          NUMACT = 0
          OLD = MEAERR
          ITEDIV = ITEDIV + 1
          NUM = MIN (ITEDIV,DIVITE)
          MEAERR = (MEAERR * (NUM-1) + NORERR) / NUM
C         WRITE (STR,*) "MEAERR=",MEAERR
C         CALL OGWRIT (2,STR)
          IF (ITEDIV .GE. DIVITE .AND. MEAERR .GT. OLD) THEN
              FINISH = -1
          ELSE
              GOTO 1000
          ENDIF
      ENDIF
C ----------------------------------------------------------------------
      IF (FINISH .EQ. -1) THEN
          FINOPT = 4
          FINITE = NUMITE
          CALL OGWRIT (1,"")
          WRITE (STR,'("OPTGRA: Converged: unf ITERAT=",2I4,2D11.3)')
     &    NUMITE,MAXITE,CONERR,DESNOR
          CALL OGWRIT (1,STR)
          GOTO 9999
      ENDIF
C ----------------------------------------------------------------------
      ITEDIV = 0
      ITEOPT = NUMITE
      IF (ITECOR .EQ. 0 .OR. CONCOR(NUMCON+1).LT.CONVAL(NUMCON+1)) THEN
      VARCOR = VARVAL
      CONCOR = CONVAL
      ENDIF
      IF (ITECOR .EQ. 0) ITECOR = NUMITE
C ----------------------------------------------------------------------
      OLD = MEAMER
      ITECNV = ITECNV + 1
      NUM = MIN (ITECNV,CNVITE)
      MEAMER = (MEAMER * (NUM-1) + CONCOR(NUMCON+1)) / NUM
C      WRITE (STR,*) "MEAMER=",ITECNV,NUM,MEAMER,OLD,OLD/MEAMER
C      CALL OGWRIT (-1,STR)
      IF (ITECNV .GE. CNVITE .AND. MEAMER .LT. OLD) THEN
          FINOPT = 2
          FINITE = ITEOPT
          VARVAL = VARCOR
          CONVAL = CONCOR
          CALL OGWRIT (1,"")
          WRITE(STR,'("OPTGRA: Converged: mat ITERAT=",2I4,2D11.3)')
     &    NUMITE,MAXITE,CONERR,DESNOR
          CALL OGWRIT (1,STR)
          GOTO 9999
      ENDIF
C ======================================================================
C OPTIMISATION PART
C ----------------------------------------------------------------------
      VARSAV = VARMAX
      VARMAX = VARMAX * 10D-1
      CALL OGOPTI (VARACC, NUMEQU, FINISH, DESNOR, CALVAL)
      VARMAX = VARSAV
C ----------------------------------------------------------------------
      IF (FINISH .EQ. 0) GOTO 1000
C ======================================================================
C NOT CONVERGED
C ----------------------------------------------------------------------
      IF (VARACC .NE. 0D0) GOTO 1000
C ======================================================================
C CONVERGED
C ----------------------------------------------------------------------
      FINOPT = 1
      FINITE = NUMITE
      CALL OGWRIT (1,"")
      WRITE (STR,'("OPTGRA: Converged: yes ITERAT=",2I4,2D11.3)')
     &NUMITE,MAXITE,CONERR,DESNOR
      CALL OGWRIT (1,STR)
      CALL OGWRIT (3,"")
C      WRITE (STR,*) "DIF=",NORM2(VARVAL-VARREF)
C      CALL OGWRIT (1,STR)
C ======================================================================
 9999 CONTINUE
C ======================================================================
C DESCALE VARIABLES
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
          VALVAR(VAR) = VARVAL(VAR) * VARSCA(VAR)
      ENDDO
C ======================================================================
C DESCALE VALUES
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON + 1
          TYP = CONTYP(CON)
          SCA = CONSCA(CON)
          IF (TYP .LT. 0) SCA = - SCA
          VALCON(CON) = CONVAL(CON) * SCA
      ENDDO
C ======================================================================
      DEALLOCATE  (VARSUM)
      DEALLOCATE  (VARCOR)
      DEALLOCATE  (CONCOR)
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      CALL OGWRIT (2,"OPTGRA END")
      CALL OGWRIT (2,"")
C ======================================================================
      RETURN
      END
