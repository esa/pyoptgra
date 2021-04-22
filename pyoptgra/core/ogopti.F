      SUBROUTINE OGOPTI (VARACC, NUMEQU, FINISH, DESNOR, CALVAL)
C ======================================================================
C OPTIMISATION PART
C ======================================================================
C I/O | VARACC           | R*8 | ITERATION SCALED DISTANCE ACCUMULATED
C ----------------------------------------------------------------------
C INP | NUMEQU           | I*4 | NUMBER OF EQUALITY CONSTRAINTS
C ----------------------------------------------------------------------
C OUT | FINISH           | I*4 | 0=LIMIT 1=OPTIM
C ----------------------------------------------------------------------
C INP | CALVAL           | EXT | FUNCTION FOR VALUES
C     |                  |     | CALDER (VARVAL, CONVAL)
C ======================================================================
C SUBROUTINES CALLED: INVRGT, INVLFT, ACTEXC, ACTINC, OGEVAL
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      REAL(8)      VARACC
C ----------------------------------------------------------------------
      INTEGER(4)   NUMEQU
      INTEGER(4)   FINISH
      EXTERNAL     CALVAL
C ======================================================================
      INTEGER(4)   STAFLG, FACCNT, NUMCOR
      REAL(8)      DESNOR, FOLDIS, COSIMP, CORNOR, QUACOR, REFDIS
      REAL(8)      CO0, CO1, CO2, NOR
      REAL(8)      COSCO2, COSCO1
      REAL(8)      MAXDIS, NORPRV
C ----------------------------------------------------------------------
      INTEGER(4)   CON, VAR, COS, ACT, IND, LEN, INC, FLG, NUM, III
      INTEGER(4)   NNN, COR, TYP, DES, PRV, MET
      REAL(8)      DER, VAL, MAX, DET, AAA, BBB, CCC, DIS, PER
      REAL(8)      FAC, DEL, REF, RAT, EXC, EPS, IMP
      REAL(8)      DIF, SCA, SAV, BET, THT
      CHARACTER    STR*256, NAM*256
C ======================================================================
      REAL(8),     DIMENSION(:), ALLOCATABLE :: COSACT
      REAL(8),     DIMENSION(:), ALLOCATABLE :: VARVEC
      REAL(8),     DIMENSION(:), ALLOCATABLE :: VARWRK
      REAL(8),     DIMENSION(:), ALLOCATABLE :: CORVEC
      REAL(8),     DIMENSION(:), ALLOCATABLE :: DESDER
      REAL(8),     DIMENSION(:), ALLOCATABLE :: DESPRV
      REAL(8),     DIMENSION(:), ALLOCATABLE :: VARPRV
      REAL(8),     DIMENSION(:), ALLOCATABLE :: CONVEC
      REAL(8),     DIMENSION(:), ALLOCATABLE :: CONQUA
      INTEGER(4),  DIMENSION(:), ALLOCATABLE :: CONCOR
C ----------------------------------------------------------------------
      ALLOCATE    (COSACT(NUMCON))
      ALLOCATE    (VARVEC(NUMVAR))
      ALLOCATE    (VARWRK(NUMVAR))
      ALLOCATE    (CORVEC(NUMCON))
      ALLOCATE    (DESDER(NUMVAR))
      ALLOCATE    (DESPRV(NUMVAR))
      ALLOCATE    (VARPRV(NUMVAR))
      ALLOCATE    (CONVEC(NUMCON+1))
      ALLOCATE    (CONQUA(NUMCON+1))
      ALLOCATE    (CONCOR(NUMCON+1))
C ======================================================================
C OPTIMISATION PART
C ----------------------------------------------------------------------
      COS = NUMCON + 1
      DES = NUMCON + 2
      PRV = NUMCON + 3
      FACCNT = 0
      CONRED(1:COS,:) = CONDER(1:COS,:)
      CONRED(  DES,:) = VARDES
      DESDER          = VARDES
      CONRED(  PRV,:) = FUNVAR
      CONDER(  PRV,:) = FUNVAR
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"OPTIMISATION PART")
C ----------------------------------------------------------------------
C      WRITE (STR,'("NUMACT = ",I4)') NUMACT
C      CALL OGWRIT (3,STR)
      NUMCOR = NUMACT
      CONCOR = ACTCON
      NUMACT = 0
      CONACT(1:DES) = 0
C      DO COR = 1, NUMCOR
C          CON = CONCOR(COR)
C          NAM = CONSTR(CON)
C          LEN = CONLEN(CON)
C          WRITE (STR,'("ACT = ",I4,5X,1X,A)') CON, NAM(1:LEN)
C          CALL OGWRIT (3,STR)
C          CALL OGINCL (CON)
C      ENDDO
C ======================================================================
 1100 CONTINUE
C ======================================================================
C VECTOR OF STEEPEST ASCENT
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"VECTOR OF STEEPEST ASCENT")
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"REMOVE AND INCLUDE CONSTRAINTS:")
      CALL OGWRIT (3,"")
      CALL OGWRIT (3," REM  INC CONSTRAINT")
C ----------------------------------------------------------------------
C REMOVE PASSIVE INEQUALITY CONSTRAINTS
C ----------------------------------------------------------------------
      DO ACT = NUMACT, 1, -1
          CON = ACTCON(ACT)
          IF (CONVAL(CON) .LE. 1D0) CYCLE
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'(I4,5X,1X,A)') CON, NAM(1:LEN)
          CALL OGWRIT (2,STR)
          CALL OGEXCL (ACT)
          CONACT(CON) = -1
      ENDDO
C ----------------------------------------------------------------------
C INCLUDE VIOLATED INEQUALITY CONSTRAINTS AND SELECT PASSIVE ONES
C SELECT PASSIVE INEQUALITY CONSTRAINTS
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON
          IF (CONTYP(CON) .EQ. -2) CYCLE
          IF     (CONACT(CON) .GT.  0  ) THEN
          ELSEIF (CONTYP(CON) .EQ.  0  ) THEN
              CONACT(CON) =  0
          ELSEIF (CONVAL(CON) .LT. -1D0) THEN
              CONACT(CON) =  0
          ELSEIF (CONVAL(CON) .LE. +1D0) THEN
              CONACT(CON) =  0
          ELSE
              CONACT(CON) = -1
          ENDIF
      ENDDO
C ======================================================================
      NNN = 1
 1110 CONTINUE
      NNN = NNN + 1
      IF (NNN .GT. 999) THEN
          FINISH = 0
          WRITE (STR,*) "NNN=",NNN
          CALL OGWRIT (2,STR) 
          GOTO 9999
      ENDIF
C ======================================================================
C DERIVATIVES OF MERIT W.R.T. ACTIVE CONSTRAINTS
C ----------------------------------------------------------------------
      CALL OGRIGT (-CONRED(COS,1:NUMACT), COSACT)
      DESNOR = DSQRT(SUM(CONRED(COS,NUMACT+1:NUMVAR)**2))
C ----------------------------------------------------------------------
C CONSTRAINT REMOVAL
C ----------------------------------------------------------------------
      IND = 0
      EXC = -1D-12
      MAX = EXC
      DO ACT = 1, NUMACT
          CON = ACTCON(ACT)
          IF (CONTYP(CON) .EQ. 0) CYCLE
          VAL = COSACT(ACT)
          FAC = DOT_PRODUCT (CONRED(CON,1:NUMVAR),
     &                       CONRED(CON,1:NUMVAR))
          FAC = DSQRT(FAC)
          VAL = VAL * FAC
          IF (VAL .GE. EXC) CYCLE
          IF (VAL .GT. MAX) CYCLE
          MAX = VAL
          IND = ACT
      ENDDO
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN
          CON = ACTCON(IND)
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'(I4,5X,3(1X,D10.3),1X,A)')
     &           CON,DESNOR,MAX,VARMAX,NAM(1:LEN)
          CALL OGWRIT (3,STR)      
          CALL OGEXCL (IND)
          GOTO 1110
      ENDIF
C ----------------------------------------------------------------------
C CONSTRAINT INCLUSION
C ----------------------------------------------------------------------
      IF (DESNOR .EQ. 0D0) GOTO 3999
C ----------------------------------------------------------------------
      INC = 0
      EPS = 1D-03
      MAX = -1D10
      MAX = 0D0
C ----------------------------------------------------------------------      
      DO CON = 1, NUMCON
          IF (CONTYP(CON) .EQ. -2) CYCLE
          IF (CONACT(CON) .NE.  0) CYCLE
          DEL = DOT_PRODUCT (CONRED(CON,NUMACT+1:NUMVAR),
     &                       CONRED(COS,NUMACT+1:NUMVAR)) / DESNOR
          VAL = DABS(DEL) * VARMAX
          IF (VAL .LT. EPS) CYCLE
          FAC = DOT_PRODUCT (CONRED(CON,1:NUMVAR),
     &                       CONRED(CON,1:NUMVAR))
          FAC = DSQRT(FAC)
          DEL = DEL / FAC
          IF (DEL .LT. 0D0 .AND. DEL .LT. MAX) THEN
              MAX = DEL
              INC = CON
          ENDIF
          IF (CONTYP(CON) .NE. 0) CYCLE
          DEL = - DEL
          IF (DEL .LT. 0D0 .AND. DEL .LT. MAX) THEN
              MAX = DEL
              INC = CON
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      IF (INC .NE. 0) THEN
          CON = INC
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'(5X,I4,3(1X,D10.3),1X,A)')
     &           CON,DESNOR,MAX,VARMAX,NAM(1:LEN)
          CALL OGWRIT (3,STR)
          CALL OGINCL (INC)
          GOTO 1110
      ENDIF
C ----------------------------------------------------------------------
 3999 CONTINUE
C ----------------------------------------------------------------------
C MATCHED INEQUALITY CONSTRAINTS + STEEPEST ASCENT VECTOR NORM
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      CALL OGWRIT (3,"STATUS OF MATCHED INEQUALITY CONSTRAINTS:")
      CALL OGWRIT (3,"")
      CALL OGWRIT (3," ACT  PAS CONSTRAINT")
      DO CON = 1, NUMCON
          IF (CONTYP(CON) .EQ. -2) CYCLE
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          IF     (CONTYP(CON) .EQ.  0  ) THEN
          ELSEIF (CONACT(CON) .GT.  0  ) THEN
              WRITE (STR,'(I4,5X,1X,A)') CON,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ELSEIF (CONACT(CON) .EQ.  0  ) THEN
              WRITE (STR,'(5X,I4,1X,A)') CON,NAM(1:LEN)
              CALL OGWRIT (3,STR)
          ENDIF
      ENDDO
      CALL OGWRIT (3,"")
      WRITE (STR,'("STEEPEST ASCENT NORM: ",D13.6)') DESNOR
      CALL OGWRIT (3,STR)
      WRITE (STR,'("MAXIMUM DISTANCE....: ",D13.6)') VARMAX
      CALL OGWRIT (3,STR)
C ======================================================================
      FINISH = 0
C ======================================================================
C IF CONVERGENCE
C ----------------------------------------------------------------------
      COSIMP = DESNOR * VARMAX
      IF (DABS (COSIMP) .LE. 1D0) THEN
          FOLDIS = 0D0
          FINISH = 1
          WRITE (STR,'("FINAL...............:",1X,D13.6,'//
     &               '11X,1(1X,D10.3),1X,D16.9)')
     &                FOLDIS, COSIMP, CONVAL(COS) + COSIMP
          CALL OGWRIT (2,STR)
          GOTO 9999
      ENDIF
C ======================================================================
C IF CONSTRAINT IS HIT
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
          VAL = CONDER(COS,VAR)
          DO ACT = 1, NUMACT
              CON = ACTCON(ACT)
              VAL = VAL + CONDER(CON,VAR) * COSACT(ACT)
          ENDDO
          VARDES(VAR) = VAL
      ENDDO
C ----------------------------------------------------------------------
      IND = 0
      DIS = 1D10
      DO CON = 1, NUMCON
         IF (CONTYP(CON) .EQ. -2) CYCLE
         IF (CONACT(CON) .NE. -1) CYCLE
         VAL = DOT_PRODUCT (CONRED(CON,NUMACT+1:NUMVAR),
     &                      CONRED(COS,NUMACT+1:NUMVAR))
         IF (VAL .EQ. 0D0) CYCLE
         VAL = - CONVAL(CON) / VAL * DESNOR
         IF (VAL .LE. 0D0) CYCLE
         IF (VAL .GE. DIS) CYCLE
         DIS = VAL
         IND = CON
      ENDDO
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN
          VAL = DSQRT(SUM((VARVAL-VARREF+VARDES*DIS/DESNOR)**2))
          IF (VAL .GT. VARMAX) IND = 0
      ENDIF
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN 
          IF (CONFIX(IND) .LE. 0) THEN
          IF (VAL .GT. VARMAX * 1D-1) IND = 0
          ENDIF
      ENDIF
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN
          IMP = DIS * DESNOR
          CON = IND
          THT = 1D0
          BET = 0D0
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'( "CONSTRAINT REACHED..:",'//
     &    '1X,D13.6,11X,1(1X,D10.3),1X,D16.9,22X,1X,I4,1X,A)')
     &    DIS, IMP, CONVAL(COS) + COSIMP,
     &    CON, NAM(1:LEN)
          CALL OGWRIT (2,STR)
          VARACC = VARACC + DIS
          VARVAL = VARVAL + DIS * VARDES / DESNOR
          DO CON = 1, NUMCON + 1
             VAL = DOT_PRODUCT (CONRED(CON,NUMACT+1:NUMVAR),
     &                          CONRED(COS,NUMACT+1:NUMVAR))
             CONVAL(CON) = CONVAL(CON) + VAL * DIS / DESNOR
          ENDDO
          GOTO 1100
      ENDIF
C ======================================================================
 7766 CONTINUE
C ----------------------------------------------------------------------
      CALL OGRIGT (-CONRED(COS,1:NUMACT), COSACT)
      DO VAR = 1, NUMVAR
          VAL = CONDER(COS,VAR)
          DO ACT = 1, NUMACT
              CON = ACTCON(ACT)
              VAL = VAL + CONDER(CON,VAR) * COSACT(ACT)
          ENDDO
          DESPRV(VAR) = VAL
      ENDDO
      DESNOR = DSQRT(SUM(DESPRV**2))
      WRITE (STR,'("DESNOR=",D13.6)') DESNOR
C      CALL OGWRIT (2,STR)
C ----------------------------------------------------------------------
      CALL OGRIGT (-CONRED(PRV,1:NUMACT), COSACT)
      DO VAR = 1, NUMVAR
          VAL = CONDER(PRV,VAR)
          DO ACT = 1, NUMACT
              CON = ACTCON(ACT)
              VAL = VAL + CONDER(CON,VAR) * COSACT(ACT)
          ENDDO
          VARPRV(VAR) = VAL
      ENDDO
      NORPRV = DSQRT(SUM(VARPRV**2))
      WRITE (STR,'("NORPRV=",D13.6)') NORPRV
C      CALL OGWRIT (2,STR)
C ----------------------------------------------------------------------
      CALL OGRIGT (-CONRED(DES,1:NUMACT), COSACT)
      DO VAR = 1, NUMVAR
          VAL = DESDER(VAR)
          DO ACT = 1, NUMACT
              CON = ACTCON(ACT)
              VAL = VAL + CONDER(CON,VAR) * COSACT(ACT)
          ENDDO
          VARDIR(VAR) = VAL
      ENDDO
      NOR = DSQRT(SUM(VARDIR**2))
      WRITE (STR,'("NOR=",D13.6)') NOR
C      CALL OGWRIT (2,STR)
C ----------------------------------------------------------------------
C MET = 3: CONJUGATE GRADIENT METHOD
C MET = 2: SPETRAL CONJUGATE GRADIENT METHOD
C MET = 1: MODIFIED SPETRAL CONJUGATE GRADIENT METHOD
C MET = 0: STEEPEST DESCENT METHOD
C ----------------------------------------------------------------------
      MET = OPTMET
      THT = 1D0
      BET = 0D0
      IF     (MET .EQ. 0) THEN
      ELSEIF (MET .EQ. 1) THEN
          VARVEC = DESPRV - VARPRV
          IF (NORPRV**2 .GT. 1D-12) THEN
          THT = - DOT_PRODUCT (VARDIR,VARVEC) / NORPRV ** 2   
          BET = DESNOR ** 2 / NORPRV ** 2
          ENDIF
      ELSEIF (MET .EQ. 2) THEN
          VARVEC = DESPRV - VARPRV
          VARWRK = VARREF - VARGRD
          VAL = DOT_PRODUCT (VARWRK,VARVEC)
          FAC = DOT_PRODUCT (VARDIR,VARVEC)
          IF (DABS (VAL) .GT. 1D-12 .AND. DABS (FAC) .GT. 1D-12) THEN
          THT = - DOT_PRODUCT (VARWRK,VARWRK) / VAL
          VARWRK = - VARVEC * THT - VARWRK
          BET = DOT_PRODUCT (VARWRK,DESPRV) / FAC
          ENDIF
      ELSEIF (MET .EQ. 3) THEN
          IF (NORPRV .NE. 0D0) THEN
          THT = 1D0
          BET = DESNOR ** 2 / NORPRV ** 2  
          ENDIF
      ENDIF
C      WRITE (STR,'("THT=",D13.6)') THT
C      CALL OGWRIT (3,STR)
C      WRITE (STR,'("BET=",D13.6)') BET
C      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
      EPS = 1D-03
C      WRITE (STR,*) "THT/BET=",THT,BET
C      CALL OGWRIT (2,STR)
      VARDES = THT * DESPRV + BET * VARDIR
      DESNOR = DSQRT(SUM(VARDES**2))
      NOR = DESNOR
      DO CON = 1, NUMCON
          IF (CONTYP(CON) .EQ. -2) CYCLE
          IF (CONACT(CON) .NE.  0) CYCLE
          DEL = DOT_PRODUCT (CONDER(CON,1:NUMVAR),
     &                       VARDES(    1:NUMVAR)) / NOR
          VAL = DABS(DEL) * VARMAX
          IF (VAL .LT. EPS) CYCLE
          FAC = DOT_PRODUCT (CONDER(CON,1:NUMVAR),
     &                       CONDER(CON,1:NUMVAR))
          DEL = DEL / DSQRT (FAC)
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          TYP = CONTYP(CON)
          IF (DEL .LT. 0D0) THEN
C          CALL OGINCL (CON)
          ACT = CONACT(CON)
C          WRITE (STR,'(5X,2I4,3(1X,D10.3),1X,A)')
C     &           CON,ACT,+NOR,VAL,DEL,NAM(1:LEN)
C          CALL OGWRIT (2,STR)
          IF (ACT .EQ. 0) THEN
          BET = 0D0
          THT = 1D0
          ELSE
          GOTO 7766
          ENDIF
          ENDIF
          IF (CONTYP(CON) .NE. 0) CYCLE
          DEL = - DEL
          IF (DEL .LT. 0D0) THEN
C          CALL OGINCL (CON)
          ACT = CONACT(CON)
C          WRITE (STR,'(5X,2I4,3(1X,D10.3),1X,A)')
C     &           CON,ACT,-NOR,VAL,DEL,NAM(1:LEN)
C          CALL OGWRIT (2,STR)
          IF (ACT .EQ. 0) THEN
          BET = 0D0
          THT = 1D0
          ELSE
          GOTO 7766
          ENDIF
          DEL = DOT_PRODUCT (CONDER(CON,1:NUMVAR),
     &                       CONDER(COS,1:NUMVAR))
          VAL = DABS(DEL) * VARMAX / DESNOR
C          WRITE (STR,'(5X,2I4,3(1X,D10.3),1X,A)')
C     &           CON,TYP,-DESNOR,VAL,DEL,NAM(1:LEN)
C          CALL OGWRIT (2,STR)
          ENDIF
      ENDDO
      COSCO1 = DOT_PRODUCT (VARDES, CONDER(COS,1:NUMVAR)) / DESNOR
      IF (COSCO1 .LT. 0D0) THEN
          WRITE (STR,*) "COSCO1=",COSCO1
          CALL OGWRIT (2,STR)
          BET = 0D0
          THT = 1D0
      ENDIF
      VARDES = THT * DESPRV + BET * VARDIR
      DESNOR = DSQRT(SUM(VARDES**2))
C      WRITE (STR,*) "THT/BET=",THT,BET
C      CALL OGWRIT (2,STR)
C ======================================================================
C SECOND ORDER DERIVATIVES BY NUMERICAL DIFFERENCING
C ----------------------------------------------------------------------
 4112 CONTINUE
C ----------------------------------------------------------------------
      CALL OGWRIT (3,"")
      WRITE (STR,'("SECOND ORDER CORRECTION")')
      CALL OGWRIT (3,STR)
      VARDES = THT * DESPRV + BET * VARDIR
      DESNOR = DSQRT(SUM(VARDES**2))
C ----------------------------------------------------------------------
C MAXIMUM TRAVEL DISTANCE
C ----------------------------------------------------------------------
      DIS = VARMAX
      VARVEC = VARVAL - VARREF
      CO0 = DOT_PRODUCT (VARVEC,VARVEC) - DIS ** 2
      IF (CO0 .GE. 0D0) THEN
           DIS = 0D0
      ELSE
           CO1 = DOT_PRODUCT (VARDES,VARVEC)
           CO2 = DESNOR ** 2
           DET = CO1 ** 2 - CO0 * CO2
           DIS = (DSQRT (DET) - CO1) / CO2
      ENDIF
      DIS = DIS * DESNOR
      MAXDIS = DIS
C ======================================================================
C COMPUTE SECOND ORDER EFFECTS
C ----------------------------------------------------------------------
      NNN = 0
      IF (SENOPT .GE. +1) THEN
      DO CON = 1, NUMCON
          IF (CONTYP(CON) .EQ. -2) CYCLE
          ACT = CONACT(CON)
          IND = SENACT(CON)
          IF (ACT .EQ. -1 .AND. IND .EQ. -1) CYCLE
          IF (ACT .EQ.  0 .AND. IND .EQ.  0) CYCLE
          IF (ACT .GT.  0 .AND. IND .GT.  0) CYCLE
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'(I4,1X,I4,1X,I4,1X,A)') CON, ACT, IND, NAM(1:LEN)
          CALL OGWRIT (2,STR)
          NNN = 1
      ENDDO
      ENDIF
      IF     (SENOPT .LE.  0 .OR. NNN .EQ. 1) THEN
          FAC = VARSTP / DESNOR
          VARVEC = VARREF + VARDES * VARSTP / DESNOR
          CALL OGEVAL (VARVEC, CONVEC, 0, CONDER, CALVAL, CALVAL)
          CONQUA = MATMUL (CONDER(1:COS,       1:NUMVAR),
     &                     VARDES(             1:NUMVAR))
          CONQUA = 2D0 * (CONVEC - CONREF - CONQUA * FAC) / FAC**2
      ENDIF
      IF     (SENOPT .EQ. -1) THEN
          SENQUA = CONQUA
      ELSEIF (SENOPT .GE. +1 .AND. NNN .EQ. 0) THEN
          CONQUA = SENQUA
      ENDIF
C ======================================================================
C COMPUTE CORRECTION VECTOR
C ----------------------------------------------------------------------
      DO ACT = 1, NUMACT
         CON = ACTCON(ACT)
         CORVEC(ACT) = CONQUA(CON)
      ENDDO
      CALL OGLEFT (CORVEC, CORVEC)
C ----------------------------------------------------------------------
      CORNOR = DSQRT(SUM(CORVEC(1:NUMACT)**2)) * 0.5D0 / DESNOR / DESNOR
      CALL OGWRIT (3,"")
      WRITE (STR,'("STEEPEST ASCENT  NORM: ",D13.6)') DESNOR
      CALL OGWRIT (3,STR)
      WRITE (STR,'("ACCUMULATED  DISTANCE: ",D13.6)') VARACC
      CALL OGWRIT (3,STR)
C ======================================================================
C GET EXTREMUM
C ----------------------------------------------------------------------
      COSCO1 = DOT_PRODUCT (VARDES, CONDER(COS,1:NUMVAR)) / DESNOR
      COSCO2 = CONQUA(COS) - DOT_PRODUCT (CONRED(COS,1:NUMACT),
     &                                    CORVEC(    1:NUMACT))
      COSCO2 = COSCO2 * 0.5D0 / DESNOR / DESNOR
      WRITE (STR,*) "COSCO2/COSCO1=",COSCO2,COSCO1
      CALL OGWRIT (3,STR)
      IF (COSCO1 .LT. 0D0) THEN
      CALL OGWRIT (2,STR)
      ENDIF
C ----------------------------------------------------------------------
      FOLDIS = 0D0
      QUACOR = CORNOR * FOLDIS * FOLDIS
      COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
      CALL OGWRIT (3,"")
      WRITE (STR,'(    "STEEPEST ASCENT FOLLOW",'//
     &           '  5X,"DISTANCE",'//
     &           '  1X,"CORRECTION",'//
     &           '  2X,"MERIT_DEL",'//
     &           '  6X,"MERIT_VALUE")')
      CALL OGWRIT (3,STR)
      WRITE (STR,'("INITIAL.............:",1X,D13.6,'//
     &           '  2(1X,D10.3),1X,D16.9)')
     &              FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
      CALL OGWRIT (3,STR)
C ======================================================================
      IF     (COSCO2 .LT. 0D0) THEN
          FOLDIS = - 0.5D0 * COSCO1 / COSCO2
          QUACOR = CORNOR * FOLDIS * FOLDIS
          COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
          WRITE (STR,'("MERIT MAXIMUM.......:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
          CALL OGWRIT (3,STR)
          STAFLG = 1
      ELSEIF (COSCO2 .GT. 0D0) THEN
          FOLDIS = VARMAX
          QUACOR = CORNOR * FOLDIS * FOLDIS
          COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
          WRITE (STR,'("MERIT CONVEX........:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
          CALL OGWRIT (3,STR)
          STAFLG = 2
      ELSE
          FOLDIS = VARMAX
          QUACOR = CORNOR * FOLDIS * FOLDIS
          COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
          WRITE (STR,'("MERIT LINEAR........:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
          CALL OGWRIT (3,STR)
          STAFLG = 2
      ENDIF
C ======================================================================
C IF MAXIMUM DISTANCE IS HIT
C ----------------------------------------------------------------------
      IF (FOLDIS .GT. VARMAX) THEN
          FOLDIS = VARMAX
          QUACOR = CORNOR * FOLDIS * FOLDIS
          COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
          WRITE (STR,'("MAXIMUM DISTANCE....:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
          CALL OGWRIT (3,STR)
          STAFLG = 2
      ENDIF
C ======================================================================
C IF CONVERGENCE
C ----------------------------------------------------------------------
      IF (DABS (COSIMP) .LE. 1D0) THEN
          WRITE (STR,'("FINAL...............:",1X,D13.6,'//
     &    '  2(1X,D10.3),1X,D16.9,2D11.3)')
     &      FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP, THT, BET
          CALL OGWRIT (2,STR)
          IF (THT .NE. 1D0 .OR. BET .NE. 0D0) THEN
              THT = 1D0
              BET = 0D0
              GOTO 4112
          ENDIF
          FOLDIS = 0D0
          FINISH = 1
          GOTO 9999
      ENDIF
C ======================================================================
C IF REMAINING DISTANCE IS HIT
C ----------------------------------------------------------------------
      IF (FOLDIS .GT. MAXDIS) THEN
          FOLDIS = MAXDIS
          QUACOR = CORNOR * FOLDIS * FOLDIS
          COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
          WRITE (STR,'("REMAINING DISTANCE..:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
          CALL OGWRIT (3,STR)
          STAFLG = 2
      ENDIF
C ======================================================================
C IF CONSTRAINT IS HIT
C ----------------------------------------------------------------------
      IND = 0
      DO CON = 1, NUMCON
          IF (CONTYP(CON) .EQ. -2) CYCLE
          IF (CONACT(CON) .NE. -1) CYCLE
          CO2 = CONQUA(CON) - DOT_PRODUCT (CONRED(CON,1:NUMACT),
     &                                     CORVEC(    1:NUMACT))
          CO1 = DOT_PRODUCT (CONDER(CON,1:NUMVAR),
     &                       VARDES(    1:NUMVAR))
          CO0 = CONVAL(CON) * 2D0
          IF    (CO2 .NE. 0D0) THEN
              DET = CO1 ** 2 -  CO2 * CO0
              IF (DET .LT. 0D0) CYCLE
              DET = DSQRT (DET)
              VAL = 1D10
              FAC = (-CO1 + DET) / CO2
              IF (FAC .GT. 0D0 .AND. FAC .LT. VAL) VAL = FAC
              FAC = (-CO1 - DET) / CO2
              IF (FAC .GT. 0D0 .AND. FAC .LT. VAL) VAL = FAC
          ELSEIF (CO1 .NE. 0D0) THEN
              VAL = -CO0 / CO1 * 0.5D0
          ELSE
              CYCLE
          ENDIF
          VAL = VAL * DESNOR
          IF (VAL .GT. 0D0 .AND. VAL .LT. FOLDIS) THEN
              FOLDIS = VAL
              IND = CON 
          ENDIF
      ENDDO
C ----------------------------------------------------------------------
      IF (IND .NE. 0) THEN
          QUACOR = CORNOR * FOLDIS * FOLDIS
          COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
          CON = IND
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'( "CONSTRAINT REACHED..:",'//
     &    '1X,D13.6,2(1X,D10.3),1X,D16.9,1X,I4,1X,A)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP, CON, NAM(1:LEN)
          CALL OGWRIT (3,STR)
          STAFLG = 3
      ENDIF
C ======================================================================
C UPDATE
C ----------------------------------------------------------------------
      REFDIS = FOLDIS
 8211 CONTINUE
      WRITE (STR,'("FINAL...............:",1X,D13.6,'//
     &'2(1X,D10.3),1X,D16.9)')
     &FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP
      CALL OGWRIT (3,STR)
C ----------------------------------------------------------------------
      FAC = FOLDIS / DESNOR
C ----------------------------------------------------------------------
C VARIABLE DELTA
C ----------------------------------------------------------------------
      CALL OGRIGT (CORVEC, COSACT)
      DIS = 0D0
      DO VAR = 1, NUMVAR
          VAL = 0D0
          DO ACT = 1, NUMACT
              IND = ACTCON(ACT)
              VAL = VAL - COSACT(ACT) * CONDER(IND,VAR)
          ENDDO
          VARVEC(VAR) = FAC * (VARDES(VAR) + (VAL * FAC * 0.5D0))
          DIS = DIS + VARVEC(VAR) * VARVEC(VAR)
      ENDDO
      DIS = DSQRT (DIS)
C ----------------------------------------------------------------------
      WRITE (STR,*) "REFDIS=",REFDIS
      CALL OGWRIT (3,STR)
      WRITE (STR,*) "FOLDIS=",FOLDIS
      CALL OGWRIT (3,STR)
      WRITE (STR,*) "DIS=",DIS
      CALL OGWRIT (3,STR)
      IF (DIS .GT. REFDIS * 1.2D0 .AND. SENOPT .GT. 0) THEN
          FACCNT = FACCNT + 1
          IF (FACCNT .LT. 10) THEN
              FOLDIS = FOLDIS * 0.5D0
              QUACOR = CORNOR * FOLDIS * FOLDIS
              COSIMP = FOLDIS * (COSCO1 + FOLDIS * COSCO2)
              GOTO 8211
          ELSE
              GOTO 9999
          ENDIF
      ENDIF
C ----------------------------------------------------------------------
C UPDATE VARIABLES
C ----------------------------------------------------------------------
      VARACC = VARACC + FOLDIS
      VARVAL = VARVAL + VARVEC
      CCC = DSQRT(SUM((VARVAL - VARREF)**2)) - VARMAX ** 2
      IF (CCC .GE. 0D0) THEN
          WRITE (STR,*) "CCC > 0",CCC
          CALL OGWRIT (3,STR)
          STAFLG = 2
      ENDIF
C ======================================================================
C MAXIMUM REACHED: NEXT ITERATION
C ----------------------------------------------------------------------
      IF (STAFLG .EQ. 1) THEN
          WRITE (STR,'("MERIT MAXIMUM.......:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9,2D11.3)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP, THT, BET
          CALL OGWRIT (2,STR)
          IF (SENOPT .GT. 0) FINISH = 1
          GOTO 9999
      ENDIF
C ======================================================================
C MAXIMUM TRAVEL DISTANCE REACHED: NEXT ITERATION
C ----------------------------------------------------------------------
      IF (STAFLG .EQ. 2) THEN
          WRITE (STR,'("REMAINING DISTANCE..:",1X,D13.6,'//
     &    '2(1X,D10.3),1X,D16.9,2D11.3)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP, THT, BET
          CALL OGWRIT (2,STR)
          GOTO 9999
      ENDIF
C ======================================================================
C CONSTRAINT HIT: UPDATE CONSTRAINT + CORRECT
C ----------------------------------------------------------------------
      IF (STAFLG .EQ. 3) THEN
          NAM = CONSTR(CON)
          LEN = CONLEN(CON)
          WRITE (STR,'( "CONSTRAINT REACHED..:",'//
     &    '1X,D13.6,2(1X,D10.3),1X,D16.9,2D11.3,1X,I4,1X,A)')
     &    FOLDIS, QUACOR, COSIMP, CONVAL(COS) + COSIMP, THT, BET,
     &    CON, NAM(1:LEN)
          CALL OGWRIT (2,STR)
          CONVEC = CONQUA - MATMUL (CONRED(1:COS,1:NUMACT),
     &                              CORVEC(      1:NUMACT))
          CONVEC = CONVEC * FAC * 0.5D0
          CONVEC = CONVEC + MATMUL (CONDER(1:COS,1:NUMVAR),
     &                              VARDES(      1:NUMVAR))
          CONVAL = CONVAL + CONVEC * FAC
          GOTO 1100
      ENDIF
C ======================================================================
 9999 CONTINUE
C ----------------------------------------------------------------------
      FUNVAR = DESPRV
      CONFIX = CONACT(1:NUMCON)
      IF (SENOPT .EQ. -1) THEN
      SENACT = CONACT(1:NUMCON)
      ENDIF
C ----------------------------------------------------------------------    
      DEALLOCATE  (COSACT)
      DEALLOCATE  (VARVEC)
      DEALLOCATE  (VARWRK)
      DEALLOCATE  (CORVEC)
      DEALLOCATE  (DESDER)
      DEALLOCATE  (DESPRV)
      DEALLOCATE  (VARPRV)
      DEALLOCATE  (CONVEC)
      DEALLOCATE  (CONQUA)
      DEALLOCATE  (CONCOR)
C ======================================================================
      RETURN
      END
