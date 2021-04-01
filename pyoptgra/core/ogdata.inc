C ======================================================================
C INCLUDE FILE OGDATA
C ======================================================================
C CONTAINS PARAMETERS AND COMMON DATA OF THE OPTIMISATION
C ======================================================================
      CHARACTER*100 SCCS_INFO_optdat(2)
     &/'%Z%%P% ',
     &'%I%	%E%	MAN_FDD\n'/
C ======================================================================
      INTEGER*4, PARAMETER :: MAXSTR=80
C ======================================================================
      INTEGER*4                                NUMVAR
      REAL*8,       DIMENSION(:  ), POINTER :: VARVAL
      INTEGER*4,    DIMENSION(:  ), POINTER :: VARTYP
      REAL*8,       DIMENSION(:  ), POINTER :: VARSCA
      CHARACTER*80, DIMENSION(:  ), POINTER :: VARSTR
      INTEGER*4,    DIMENSION(:  ), POINTER :: VARLEN
      REAL*8,       DIMENSION(:  ), POINTER :: VARREF
      REAL*8,       DIMENSION(:  ), POINTER :: VARDES
      REAL*8,       DIMENSION(:  ), POINTER :: VARGRD
      REAL*8,       DIMENSION(:  ), POINTER :: VARDIR
      REAL*8,       DIMENSION(:  ), POINTER :: FUNVAR
C ----------------------------------------------------------------------
      INTEGER*4                                NUMCON
      REAL*8,       DIMENSION(:  ), POINTER :: CONVAL
      INTEGER*4,    DIMENSION(:  ), POINTER :: CONTYP
      INTEGER*4,    DIMENSION(:  ), POINTER :: CONPRI
      REAL*8,       DIMENSION(:  ), POINTER :: CONSCA
      CHARACTER*80, DIMENSION(:  ), POINTER :: CONSTR
      INTEGER*4,    DIMENSION(:  ), POINTER :: CONLEN
      REAL*8,       DIMENSION(:  ), POINTER :: CONREF
C ----------------------------------------------------------------------
      INTEGER*4                                OPTMET
      INTEGER*4                                MAXITE
      INTEGER*4                                CORITE
      INTEGER*4                                OPTITE
	  INTEGER*4                                DIVITE
	  INTEGER*4                                CNVITE
	  REAL*8                                   VARMAX
      REAL*8                                   VARSND
      REAL*8                                   VARSTP
C ----------------------------------------------------------------------
      INTEGER*4                                VARDER
      REAL*8,       DIMENSION(:  ), POINTER :: VARPER
C ----------------------------------------------------------------------
      INTEGER*4                                LOGLUN
      INTEGER*4                                LOGLEV
C ----------------------------------------------------------------------
      INTEGER*4                                MATLEV
C ----------------------------------------------------------------------
      INTEGER*4                                TABLUN
      INTEGER*4                                TABLEV
C ----------------------------------------------------------------------
      INTEGER*4                                GRDCHK
C ----------------------------------------------------------------------
      INTEGER*4                                NUMACT
      INTEGER*4,    DIMENSION(:  ), POINTER :: ACTCON
	  INTEGER*4,    DIMENSION(:  ), POINTER :: CONFIX
	  INTEGER*4,    DIMENSION(:  ), POINTER :: CONACT
      REAL*8,       DIMENSION(:,:), POINTER :: CONDER
      REAL*8,       DIMENSION(:,:), POINTER :: CONRED
      INTEGER*4                                CONVER
      INTEGER*4,    DIMENSION(:  ), POINTER :: CONOPT
C ======================================================================
      COMMON /OGDINT/ NUMVAR, VARTYP, VARLEN,
     &                NUMCON, CONTYP, CONPRI, CONLEN,
     &                OPTMET, MAXITE, CORITE, OPTITE, DIVITE, CNVITE,
     &                VARDER,
     &                LOGLUN, LOGLEV, MATLEV, TABLUN, TABLEV, GRDCHK,
     &                NUMACT, ACTCON, CONFIX,
     &                CONACT, CONVER, CONOPT
C ======================================================================
      COMMON /OGDREA/ VARMAX, VARSND, VARSTP, VARVAL, VARSCA, VARREF,
     &                VARDES, VARGRD, VARDIR, FUNVAR, VARPER,
     &                CONVAL, CONSCA, CONREF,
     &                CONDER, CONRED
C ======================================================================
      COMMON /OGDSTR/ VARSTR, CONSTR
C ======================================================================
