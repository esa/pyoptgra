      SUBROUTINE OGCORR (VARACC, FINISH, TOTERR, NORERR, CALVAL, CALDER)
C ======================================================================
C CORRECTION PART
C ======================================================================
C SUBROUTINES CALLED: OGRIGT, OGLEFT, OGEXCL, OGINCL, OGEVAL
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
      CHARACTER*100 SCCS_INFO(2)
      DATA SCCS_INFO
     &/'%Z%%P% ',
     &'%I%      %E%     MAN_FDD\n'/
C ======================================================================
#include 'ogdata.inc'
C ======================================================================
      REAL*8        VARACC
      INTEGER*4     FINISH
      EXTERNAL      CALVAL, CALDER
C ======================================================================
      INTEGER*4     CORITR, NUMFFF, NUMCOR, MINPRI, MAXPRI, CURPRI
      REAL*8        CORNOR, FOLDIS, CSTVAL, REFNOR, SAVNOR, VARNOR
      REAL*8        CONCO0, CONCO1, CONERR, TOTERR, NORERR
      REAL*8        CORINV, VARVIO, CONMAX, NORMAX
      INTEGER*4     CONIND, NORIND, INELOP
C ----------------------------------------------------------------------
      INTEGER*4     CON, VAR, ACT, IND, LEN, COS, INC, STP, FND
      INTEGER*4     TYP, COR, QUA, NNN, LLL, CMP, PRI, VIO, FFF
      REAL*8        VAL, FAC, UPR, DEL, SIG, CO2, CO1, CO0, DE2, DIS
      REAL*8        EPS, NEW, OLD, ERR, DLT, NOR, DDD, SCA, DIF
      REAL*8        EXC
      CHARACTER*256 STR, NAM
C ======================================================================
      REAL*8,       DIMENSION(:), ALLOCATABLE :: COSACT
      REAL*8,       DIMENSION(:), ALLOCATABLE :: VARVEC
      REAL*8,       DIMENSION(:), ALLOCATABLE :: VARSAV
	REAL*8,       DIMENSION(:), ALLOCATABLE :: VARCOR
      REAL*8,       DIMENSION(:), ALLOCATABLE :: CORVEC
      REAL*8,       DIMENSION(:), ALLOCATABLE :: CONSAV
	INTEGER*4,    DIMENSION(:), ALLOCATABLE :: CONTTT
      INTEGER*4,    DIMENSION(:), ALLOCATABLE :: CONCOR
      INTEGER*4,    DIMENSION(:), ALLOCATABLE :: CONINC
      INTEGER*4,    DIMENSION(:), ALLOCATABLE :: CONHIT
      INTEGER*4,    DIMENSION(:), ALLOCATABLE :: FFFCON
C ----------------------------------------------------------------------
      ALLOCATE     (COSACT(NUMVAR))
      ALLOCATE     (VARVEC(NUMVAR))
      ALLOCATE     (VARSAV(NUMVAR))
	ALLOCATE     (VARCOR(NUMVAR))
      ALLOCATE     (CORVEC(NUMVAR))
      ALLOCATE     (CONSAV(NUMCON+1))
	ALLOCATE     (CONTTT(NUMCON+1))
      ALLOCATE     (CONCOR(NUMCON))
      ALLOCATE     (CONINC(NUMCON))
      ALLOCATE     (CONHIT(NUMCON))
      ALLOCATE     (FFFCON(NUMCON))
C ======================================================================
C CORRECTION PART
C ----------------------------------------------------------------------
      CORITR = 0
      INELOP = 0
C ======================================================================
      COS = NUMCON + 1
      VIO = NUMCON + 2
      STP = 0
      FINISH = 0
      EPS = 1D-03
      DLT = 1D-06
      VARVIO = VARMAX * 1D+03
C ----------------------------------------------------------------------
      MINPRI =  1000
      MAXPRI = -1000
      DO CON = 1, NUMCON
          MINPRI = MIN (MINPRI, CONPRI(CON))
          MAXPRI = MAX (MAXPRI, CONPRI(CON))
      ENDDO
      CONTTT = CONTYP
      CONCOR = 0
C ----------------------------------------------------------------------
      NUMFFF = NUMACT
      FFFCON = ACTCON
      FFFCON = 0
C ======================================================================
 1000 CONTINUE ! Evaluation loop
C ======================================================================
C CHECK CONSTRAINTS
C ----------------------------------------------------------------------
      NUMACT = 0
      CONACT = 0
      CONRED(1:NUMCON+1,:) = CONDER(1:NUMCON+1,:)
      CONRED(  NUMCON+2,:) = VARDIR
C ----------------------------------------------------------------------
      CONERR = 0D0
      NORERR = 0D0
      CONIND = 0
      NORIND = 0
      CONMAX = 0D0
      NORMAX = 0D0
      DO CON = 1, NUMCON
          TYP = CONTYP(CON)
          VAL = CONVAL(CON)
          IF     (VAL .LT. -1D0) THEN
              ERR = DABS (VAL)
          ELSEIF (TYP .NE.  0  ) THEN
              ERR = 0D0
          ELSEIF (VAL .GT.  1D0) THEN
              ERR = DABS (VAL)
          ELSE
              ERR = 0D0
          ENDIF
          CONERR = CONERR + ERR
          IF (ERR .GT. CONMAX) THEN
              CONIND = CON
              CONMAX = ERR
          ENDIF
          FAC = 0D0
          DO VAR = 1, NUMVAR
              FAC = FAC + CONDER(CON,VAR) ** 2
          ENDDO
          FAC = DSQRT (FAC)
          IF     (ERR .EQ.  0D0) THEN
          ELSEIF (FAC .NE.  0D0) THEN
             ERR = ERR / FAC
          ELSE
              CALL OGWRIT (0,"")
              CALL OGWRIT (0,"ERROR: CONSTRAINT CAN NOT BE SATISFIED")
              WRITE (STR,'("CON/VAL= ",I5,1X,D13.6)') CON,VAL
              CALL OGWRIT (0,STR)
              STOP
          ENDIF
          NORERR = NORERR + ERR
          IF (ERR .GT. NORMAX) THEN
              NORIND = CON
              NORMAX = ERR
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      TOTERR = CONERR
      CALL OGWRIT (3,"")
      WRITE (STR,'("NUMFFF/TOTERR/NORERR/COSVAL=",I4,3(1X,D13.6))')
     &              NUMFFF,TOTERR,NORERR,CONVAL(NUMCON+1)
      CALL OGWRIT (2,STR)
      CALL OGWRIT (3,"")
      WRITE (STR,'("MAXIM TOTAL ERROR.: ",D13.6,I6)') CONMAX, CONIND
      CALL OGWRIT (3,STR)
      WRITE (STR,'("MAXIM NORM  ERROR.: ",D13.6,I6)') NORMAX, NORIND
      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
      IF     (CORITR .EQ. 0 .AND. CONERR .EQ. 0D0) THEN
          FINISH = 1
          GOTO 9999
      ELSEIF (CORITR .NE. 0 .AND. CONERR .EQ. 0D0) THEN
          FINISH = 1
          GOTO 9999
      ELSEIF (CORITR .EQ. 10) THEN
          FINISH = 0
          CALL OGWRIT (3,"")
          WRITE (STR,'("CORITR=",I2)') CORITR
          CALL OGWRIT (3,STR)
          GOTO 9999
      ELSE
          FINISH = 0
          CORITR = CORITR + 1
      ENDIF
C ----------------------------------------------------------------------
      CURPRI = MINPRI
C ======================================================================
 2000 CONTINUE ! Priority loop
C ----------------------------------------------------------------------
      INELOP = 1
C ======================================================================
 3000 CONTINUE ! Inequality loop
C ----------------------------------------------------------------------
      VARSAV = VARVAL
      CONSAV = CONVAL
      CONTYP = CONTTT
      IF (INELOP .GT. 1) THEN
	   DO FFF = 1, NUMFFF
	       CON = FFFCON(FFF)
             CONTYP(CON) = 0
         ENDDO
      ENDIF
C ======================================================================
C MINIMUM NORM CORRECTION
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'("CORRECTION OF CONSTRAINTS")')
      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
      CONERR = 0D0
      DO CON = 1, NUMCON
          CONHIT(CON) = 0
          IF     (CONVAL(CON) .LT. -EPS) THEN
              CONERR = CONERR + DABS (CONVAL(CON))
          ELSEIF (CONTYP(CON) .NE.  0  ) THEN
          ELSEIF (CONVAL(CON) .GT. +EPS) THEN
              CONERR = CONERR + DABS (CONVAL(CON))
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'("LINEAR ERROR.: ",D13.6)') CONERR
      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'(" ACT  PAS  MOV",'//
     &           ' " COST___VAL COST___GRD",'//
     &           ' " DIST___DEL CONSTRAINT")')
      CALL OGWRIT (3,STR)
C ======================================================================
 4000 CONTINUE ! move loop
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON
          CONINC(CON) = 0
          PRI = CONPRI(CON)
          VAL = CONVAL(CON)
          TYP = CONTYP(CON)
          ACT = CONACT(CON)
          COR = CONCOR(CON)
          IF     (ACT .GT.  0  ) CYCLE
          IF     (PRI .GT. CURPRI) THEN
              CONACT(CON) = -1
          ELSEIF (VAL .LT. -DLT) THEN
              CONACT(CON) = -1
          ELSEIF (VAL .GT. +DLT) THEN
              CONACT(CON) = -1
          ELSE
              CONACT(CON) =  0
          ENDIF
          IF (PRI .GT. CURPRI) THEN
              CONCOR(CON) =  0
          ELSEIF (VAL .LT. -DLT) THEN
              CONCOR(CON) = -1
          ELSEIF (TYP .NE.  0) THEN
              CONCOR(CON) =  0
          ELSEIF (VAL .GT. +DLT) THEN
              CONCOR(CON) = +1
          ELSE
              CONCOR(CON) =  0
          ENDIF
          IF (ACT .NE. CONACT(CON) .OR. COR .NE. CONCOR(CON)) THEN
              NAM = CONSTR(CON)
              LEN = CONLEN(CON)
              WRITE (STR,'(5X,5X,I4,23X,D10.3,1X,A,4I4)')
     &        CON, CONVAL(CON),NAM(1:LEN),
     &        CONACT(CON),CONCOR(CON), ACT, COR
              CALL OGWRIT (3,STR)
          ENDIF
      ENDDO
C ======================================================================
C STEEPEST ASCENT VECTOR
C ======================================================================
C MERIT VALUE AND DERIVATIVES
C ----------------------------------------------------------------------
      CSTVAL = 0D0
      VARVEC = 0D0
      CONRED(VIO,:) = 0D0
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON
          IF (CONPRI(CON) .GT. CURPRI) CYCLE
          IF (CONCOR(CON) .EQ. 0) CYCLE
          FAC = CONCOR(CON)
          CSTVAL = CSTVAL - CONVAL(CON) * FAC
          CONRED(VIO,:) = CONRED(VIO,:) - CONRED(CON,:) * FAC
          VARVEC        = VARVEC        - CONDER(CON,:) * FAC
      ENDDO
C ----------------------------------------------------------------------        
 5000 CONTINUE
C ----------------------------------------------------------------------
C STEEPEST ASCENT VECTOR
C ----------------------------------------------------------------------
      CORVEC = CONRED(VIO,:)
C ----------------------------------------------------------------------
      CORNOR = DSQRT(SUM(CORVEC(NUMACT+1:NUMVAR)**2))
C ----------------------------------------------------------------------
C MERIT PARTIAL W.R.T. CONSTRAINTS
C ----------------------------------------------------------------------
      CALL OGRIGT (CORVEC, COSACT)
C ----------------------------------------------------------------------
C CONSTRAINT REMOVAL
C ----------------------------------------------------------------------
      IND = 0
      EXC = 1D-12
      UPR = EXC
      DO ACT = 1, NUMACT
          CON = ACTCON(ACT)
          IF (CONTYP(CON) .EQ. 0) CYCLE
          VAL = COSACT(ACT)
          IF (VAL .LE. EXC) CYCLE
          IF (VAL .LT. UPR) CYCLE
C         IF (VAL .GE. UPR .AND. UPR .GT. 0D0) CYCLE
          UPR = VAL
          IND = ACT
      ENDDO
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN
          CON = ACTCON(IND)
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'(5X,I4,5X,3(1X,D10.3),1X,A)')
     &    CON, CSTVAL, CORNOR, UPR, NAM(1:LEN)
          CALL OGWRIT (3,STR)
          CALL OGEXCL (IND)
          IF (CONINC(CON) .GE. 5) THEN
              WRITE (STR,'("OGCORR-WARNING: CONSTRAINT INCLUDED")')
              CALL OGWRIT (1,STR)
              WRITE (STR,'("CON/INC/UPR=",2I4,1X,D10.3)')
     &        CON, CONINC(CON), UPR
              CALL OGWRIT (1,STR)
          ENDIF
          IF (CONINC(CON) .GE. 20) THEN
              FINISH = 0
              GOTO 6677
          ENDIF
          GOTO 5000
      ENDIF
C ----------------------------------------------------------------------
C NORMALISE STEEPEST ASCEND VECTOR
C ----------------------------------------------------------------------
      IF (CSTVAL .LT. -CORNOR*VARVIO) THEN
          GOTO 1190
      ENDIF
C ----------------------------------------------------------------------
      CORINV = 1D0 / CORNOR
	CORVEC(NUMACT+1:NUMVAR) = CORVEC(NUMACT+1:NUMVAR) * CORINV
C ----------------------------------------------------------------------
C CONSTRAINT INCLUSION
C ----------------------------------------------------------------------
      IND = 0
      UPR = 0D0
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON
          IF (CONPRI(CON) .GT. CURPRI) CYCLE
          IF (CONACT(CON) .NE. 0) CYCLE
          DEL = DOT_PRODUCT (CONRED(CON,NUMACT+1:NUMVAR),
     &                       CONRED(VIO,NUMACT+1:NUMVAR))
          VAL = DABS(DEL) * VARMAX / CORNOR
          IF (VAL .LT. EPS) CYCLE 
          FAC = DOT_PRODUCT (CONRED(CON,       1:NUMVAR),
     &                       CONRED(CON,       1:NUMVAR))
          DEL = DEL / DSQRT (FAC)
          IF (DEL .LT. UPR) THEN
              UPR = DEL
              IND = CON
          ENDIF
          IF (CONTYP(CON) .NE. 0) CYCLE
          IF (CONCOR(CON) .NE. 0) CYCLE
          DEL = - DEL
          IF (DEL .LT. UPR) THEN
              UPR = DEL
              IND = CON
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN
          CON = IND
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'(I4,5X,5X,3(1X,D10.3),1X,A)')
     &    CON, CSTVAL, CORNOR, UPR, NAM(1:LEN)
          CALL OGWRIT (3,STR)
          CALL OGINCL (IND)
          CONINC(CON) = CONINC(CON)+1
          GOTO 5000
      ENDIF
C ----------------------------------------------------------------------
 1190 CONTINUE
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
          VAL = VARVEC(VAR)
          DO ACT = 1, NUMACT
              CON = ACTCON(ACT)
              VAL = VAL - CONDER(CON,VAR) * COSACT(ACT)
          ENDDO
          VARVEC(VAR) = VAL * CORINV
	ENDDO
C ----------------------------------------------------------------------
	VARCOR = VARVAL - VARREF
      CO2 = DOT_PRODUCT (VARVEC, VARVEC)
      CO1 = DOT_PRODUCT (VARVEC, VARCOR) * 0.5D0
      CO0 = DOT_PRODUCT (VARCOR, VARCOR) - VARMAX ** 2
      DE2 = CO1 ** 2 - CO2 * CO0
      IF (DE2 .GE. 0D0 .AND. CO2 .NE. 0D0) THEN
          DIS = (DSQRT (DE2) - CO1) / CO2
      ELSE
          DIS = 0D0
      ENDIF
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
          FAC = VARVEC(VAR)
          IF (FAC .EQ. 0D0) CYCLE
          DIF = VARVAL(VAR) - VARREF(VAR)
          SCA = VARMAX * 1D-0
          VAL = (DIF + SCA) / FAC
          FAC = (DIF - SCA) / FAC
          IF (FAC .GT. VAL) VAL = FAC
          IF (VAL .LT. DIS) DIS = VAL
      ENDDO
      IF (DIS .LT. 0D0) DIS = 0D0
C ----------------------------------------------------------------------
      FOLDIS = DIS
C ======================================================================
C OPTIMISE DIRETION OF STEPPEST ASCENT
C ======================================================================
      IF (CSTVAL .EQ. 0D0) THEN
C ----------------------------------------------------------------------
          WRITE (STR,'("CNV=",3(1X,D10.3))') CSTVAL/CORNOR/VARVIO
          CALL OGWRIT (3,STR)
          WRITE (STR,'(4X,5X,5X,3(1X,D10.3))') CSTVAL, CORNOR, FOLDIS
          CALL OGWRIT (3,STR)
          CALL OGWRIT (3,"")
          IF     (INELOP .EQ. 1) THEN
              WRITE (STR,'("INELOP===",I3)') INELOP
              CALL OGWRIT (3,STR)
              INELOP = INELOP + 1
              GOTO 3000
          ELSEIF (CURPRI .GE. MAXPRI) THEN
              WRITE (STR,'("MAXPRI===",I3)') MAXPRI
              CALL OGWRIT (3,STR)
              GOTO 5555
          ELSE
              WRITE (STR,'("CURPRI===",I3)') CURPRI
              CALL OGWRIT (3,STR)
              CURPRI = CURPRI + 1
              GOTO 2000
          ENDIF
C ----------------------------------------------------------------------
      ELSEIF (CSTVAL .GT. -CORNOR*VARVIO) THEN
C ----------------------------------------------------------------------
      ELSE
C ----------------------------------------------------------------------
          WRITE (STR,'("CNV=",3(1X,D10.3))') CSTVAL/CORNOR/VARVIO
          CALL OGWRIT (2,STR)
          CALL OGWRIT (3,"")
          WRITE (STR,'("CSTVAL=",3D10.3)') CSTVAL,CORNOR,VARVIO
          CALL OGWRIT (3,STR)
      	FINISH = 0
          GOTO 6677
          
      	IF (CORITR .LE. 1) THEN
          GOTO 5555
		ELSE
      	FINISH = 0
          GOTO 6677
          ENDIF    
C          IF     (INELOP .EQ. 1) THEN
C              WRITE (STR,'("INELOP==",I3)') INELOP
C              CALL OGWRIT (3,STR)
C              INELOP = INELOP + 1
C              GOTO 5555
C          ELSEIF (CURPRI .GE. MAXPRI) THEN
C              WRITE (STR,'("MAXPRI==",I3)') MAXPRI
C              CALL OGWRIT (3,STR)
C              FINISH = -1
C              GOTO 6677
C          ELSE
C              WRITE (STR,'("CURPRI==",I3)') CURPRI
C              CALL OGWRIT (3,STR)
C              CURPRI = CURPRI + 1
C              GOTO 2000
C          ENDIF
C ----------------------------------------------------------------------          
      ENDIF
C ======================================================================
C IF CONSTRAINT IS HIT
C ----------------------------------------------------------------------
      IND = 0
      DO CON = 1, NUMCON
          IF (CONACT(CON) .NE. -1) CYCLE
          VAL = DOT_PRODUCT (CONRED(CON,NUMACT+1:NUMVAR),
     &                       CORVEC(    NUMACT+1:NUMVAR))
          IF (VAL .EQ. 0D0) CYCLE
          VAL = - CONVAL(CON) / VAL
          IF (VAL .LE. 0D0) CYCLE
          IF (VAL .GE. FOLDIS) CYCLE
          FOLDIS = VAL
          IND = CON
      ENDDO
C ======================================================================
C UPDATE VARIABLES, CONSTRAINTS AND COST FUNCTION
C ----------------------------------------------------------------------
      VARVEC = VARVEC * FOLDIS
C ----------------------------------------------------------------------
      VARACC = VARACC + FOLDIS
      VARVAL = VARVAL + VARVEC
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON + 1
          VAL = DOT_PRODUCT (CORVEC(    NUMACT+1:NUMVAR),
     &	                   CONRED(CON,NUMACT+1:NUMVAR))
          CONVAL(CON) = CONVAL(CON) + VAL * FOLDIS
      ENDDO
C ----------------------------------------------------------------------
      CSTVAL = CSTVAL + FOLDIS * CORNOR
C ======================================================================
C MAXIMUM TRAVEL DISTANCE REACHED: NEXT ITERATION
C ----------------------------------------------------------------------
      IF (IND .EQ. 0) THEN
          WRITE (STR,'("CNV=",3(1X,D10.3))') CSTVAL/CORNOR/VARVIO
          CALL OGWRIT (3,STR)
          WRITE (STR,'(4X,5X,5X,3(1X,D10.3))') CSTVAL, CORNOR, FOLDIS
          CALL OGWRIT (3,STR)
          IF     (INELOP .EQ. 1) THEN
              WRITE (STR,'("INELOP=",I3)') INELOP
              CALL OGWRIT (3,STR)
              GOTO 9999
          ELSEIF (CURPRI .GE. MAXPRI) THEN
              WRITE (STR,'("MAXPRI=",I3)') MAXPRI
              CALL OGWRIT (3,STR)
              VARVAL = VARSAV
              CONVAL = CONSAV
              NUMACT = 0
              GOTO 5555
          ELSE
              WRITE (STR,'("CURPRI=",I3)') CURPRI
              CALL OGWRIT (3,STR)
              CURPRI = CURPRI + 1
              GOTO 2000
          ENDIF
      ENDIF
C ======================================================================
C CONSTRAINT HIT: UPDATE CONSTRAINTS + CORRECT
C ----------------------------------------------------------------------
      CON = IND
      NAM = CONSTR(CON)
      LEN = CONLEN(CON)
      VAL = CONVAL(CON)
      WRITE (STR,'(5X,5X,I4,3(1X,D10.3),1X,A)')
     &CON, CSTVAL, CORNOR, FOLDIS, NAM(1:LEN)
      CALL OGWRIT (3,STR)
      IF (CONHIT(CON) .GE. 20) THEN
          WRITE (STR,'("OGCORR-WARNING: CONSTRAINT HIT")')
          CALL OGWRIT (1,STR)
          WRITE (STR,'("CON/HIT=",2I4)') CON, CONHIT(CON)
          CALL OGWRIT (1,STR)
          FINISH = 0
          GOTO 6677
      ENDIF
      CONHIT(CON) = CONHIT(CON) + 1
C ----------------------------------------------------------------------
      GOTO 4000
C ======================================================================
 5555 CONTINUE
C ======================================================================
C MATCHED INEQUALITY CONSTRAINTS + MINIMUM CORRECTION NORM
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"STATUS OF CONSTRAINTS:")
      CALL OGWRIT (3,"")
      CALL OGWRIT (3," ACT  PAS  NON COST___VAL CONSTRAINT")
      DO CON = 1, NUMCON
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          VAL = CONVAL(CON)
          IF     (CONACT(CON) .GT.  0) THEN
              WRITE (STR,'( I4,5X,6X,D10.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ELSEIF (CONACT(CON) .EQ.  0) THEN
              WRITE (STR,'( 5X,I4,6X,D10.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ELSEIF (CONACT(CON) .LT.  0) THEN
              WRITE (STR,'(10X,I4,1X,D10.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ENDIF
      ENDDO
C ======================================================================
      CALL OGEVAL (VARVAL, CONVAL, 0, CONDER, CALVAL, CALDER)
C ======================================================================
      GOTO 1000
C ======================================================================
 9999 CONTINUE
C ======================================================================
C MATCHED INEQUALITY CONSTRAINTS + MINIMUM CORRECTION NORM
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'("CSTVAL:",D13.6)') CSTVAL
      CALL OGWRIT (3,STR)
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"STATUS OF CONSTRAINTS:")
      CALL OGWRIT (3,"")
      CALL OGWRIT (3," ACT  PAS  NON COST___VAL CONSTRAINT")
      DO CON = 1, NUMCON
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          VAL = CONVAL(CON)
          IF     (CONTYP(CON) .EQ. -2) THEN
          ELSEIF (CONACT(CON) .GT.  0) THEN
              WRITE (STR,'( I4,5X,6X,D10.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ELSEIF (CONACT(CON) .EQ.  0) THEN
              WRITE (STR,'( 5X,I4,6X,D10.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ELSEIF (CONACT(CON) .LT.  0) THEN
              WRITE (STR,'(10X,I4,1X,D10.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"STATUS OF VIOLATED CONSTRAINTS:")
      CALL OGWRIT (3,"")
      CALL OGWRIT (3," CON COST___VAL CONSTRAINT")
      CONERR = 0D0
      DO CON = 1, NUMCON
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          VAL = CONVAL(CON)
          IF     (CONTYP(CON) .EQ. -2  ) THEN
          ELSEIF (VAL .LT. -DLT) THEN
              CONERR = CONERR + DABS (VAL)
              WRITE (STR,'( I4,D11.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ELSEIF (CONTYP(CON) .NE.  0  ) THEN
          ELSEIF (VAL .GT.  DLT) THEN
              CONERR = CONERR + DABS (VAL)
              WRITE (STR,'( I4,D11.3,1X,A)') CON,VAL,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'("LINEAR ERROR.: ",D13.6)') CONERR
      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
 6677 CONTINUE
C ----------------------------------------------------------------------
      CONTYP =  CONTTT
C ----------------------------------------------------------------------
      DEALLOCATE   (COSACT)
      DEALLOCATE   (VARVEC)
      DEALLOCATE   (VARSAV)
	DEALLOCATE   (VARCOR)
      DEALLOCATE   (CORVEC)
      DEALLOCATE   (CONSAV)
	DEALLOCATE   (CONTTT)
      DEALLOCATE   (CONCOR)
      DEALLOCATE   (CONINC)
      DEALLOCATE   (CONHIT)
      DEALLOCATE   (FFFCON)
C ======================================================================
      RETURN
      END
