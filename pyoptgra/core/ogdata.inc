C ======================================================================
C INCLUDE FILE OGDATA
C ======================================================================
C CONTAINS PARAMETERS AND COMMON DATA OF THE OPTIMISATION
C ======================================================================
      INTEGER(4),   PARAMETER :: MAXSTR=80
C ======================================================================
      INTEGER(4)                               NUMVAR
      REAL(8),      DIMENSION(:  ), POINTER :: VARVAL
      INTEGER(4),   DIMENSION(:  ), POINTER :: VARTYP
      REAL(8),      DIMENSION(:  ), POINTER :: VARSCA
      CHARACTER,    DIMENSION(:  ), POINTER :: VARSTR*80
      INTEGER(4),   DIMENSION(:  ), POINTER :: VARLEN
      REAL(8),      DIMENSION(:  ), POINTER :: VARREF
      REAL(8),      DIMENSION(:  ), POINTER :: VARDES
      REAL(8),      DIMENSION(:  ), POINTER :: VARGRD
      REAL(8),      DIMENSION(:  ), POINTER :: VARDIR
      REAL(8),      DIMENSION(:  ), POINTER :: FUNVAR
      REAL(8),      DIMENSION(:  ), POINTER :: SENVAR
C ----------------------------------------------------------------------
      INTEGER(4)                               NUMCON
      REAL(8),      DIMENSION(:  ), POINTER :: CONVAL
      INTEGER(4),   DIMENSION(:  ), POINTER :: CONTYP
      INTEGER(4),   DIMENSION(:  ), POINTER :: CONPRI
      REAL(8),      DIMENSION(:  ), POINTER :: CONSCA
      CHARACTER,    DIMENSION(:  ), POINTER :: CONSTR*80
      INTEGER(4),   DIMENSION(:  ), POINTER :: CONLEN
      REAL(8),      DIMENSION(:  ), POINTER :: CONREF
      REAL(8),      DIMENSION(:  ), POINTER :: SENQUA
      REAL(8),      DIMENSION(:  ), POINTER :: SENCON
      REAL(8),      DIMENSION(:  ), POINTER :: SENDEL
      INTEGER(4),   DIMENSION(:  ), POINTER :: SENACT
C ----------------------------------------------------------------------
      INTEGER(4)                               OPTMET
      INTEGER(4)                               MAXITE
      INTEGER(4)                               CORITE
      INTEGER(4)                               OPTITE
      INTEGER(4)                               DIVITE
      INTEGER(4)                               CNVITE
      REAL(8)                                  VARMAX
      REAL(8)                                  VARSND
      REAL(8)                                  VARSTP
C ----------------------------------------------------------------------
      INTEGER(4)                               VARDER
      REAL(8),      DIMENSION(:  ), POINTER :: VARPER
C ----------------------------------------------------------------------
      INTEGER(4)                               LOGLUN  ! log file unit
      INTEGER(4)                               LOGLEV  ! log level
C ----------------------------------------------------------------------
      INTEGER(4)                               LOGLUP  ! pygmo log file unit
      INTEGER(4)                               VERBOS  ! pygmo verbosity
      INTEGER(4)                               FEVALS  ! pygmo: number of const fun evals
      INTEGER(4)                               PYGFLA  ! pygmo: flag indicating status of optimisation
      INTEGER(4)                               NUMITE  ! number of iterations
C ----------------------------------------------------------------------
      INTEGER(4)                               MATLEV
C ----------------------------------------------------------------------
      INTEGER(4)                               TABLUN
      INTEGER(4)                               TABLEV
C ----------------------------------------------------------------------
      INTEGER(4)                               SENOPT
C ----------------------------------------------------------------------
      INTEGER(4)                               NUMACT
      INTEGER(4),   DIMENSION(:  ), POINTER :: ACTCON
      INTEGER(4),   DIMENSION(:  ), POINTER :: CONFIX
      INTEGER(4),   DIMENSION(:  ), POINTER :: CONACT
      REAL(8),      DIMENSION(:,:), POINTER :: CONDER
      REAL(8),      DIMENSION(:,:), POINTER :: CONRED
      REAL(8),      DIMENSION(:,:), POINTER :: SENDER
      INTEGER(4)                               CONVER
      INTEGER(4),   DIMENSION(:  ), POINTER :: CONOPT
C ======================================================================
      COMMON /OGDINT/ NUMVAR, NUMCON, VARTYP, VARLEN,
     &                CONTYP, CONPRI, CONLEN, SENACT,
     &                OPTMET, MAXITE, CORITE, OPTITE, DIVITE, CNVITE,
     &                VARDER,
     &                LOGLUN, LOGLEV, MATLEV, TABLUN, TABLEV, SENOPT,
     &                NUMACT, CONVER, ACTCON, CONFIX,
     &                CONACT, CONOPT, LOGLUP, VERBOS, FEVALS, PYGFLA,
     &                NUMITE
C ======================================================================
      COMMON /OGDREA/ VARMAX, VARSND, VARSTP, VARVAL, VARSCA, VARREF,
     &                VARDES, VARGRD, VARDIR, FUNVAR, SENVAR, VARPER,
     &                CONVAL, CONSCA, CONREF, SENQUA, SENCON, SENDEL,
     &                CONDER, CONRED, SENDER
C ======================================================================
      COMMON /OGDSTR/ VARSTR, CONSTR
C ======================================================================
