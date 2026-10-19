"""Legacy models and records of the MRI reference, kept so that their checks still run (tests/legacy/):

    export, stream, pine_gen      F3_g (sessions 1-2): export JSON / golden, bar-by-bar stream with EMA, Pine generator
    calibration                   Platt calibration of F3_g and NT3_PRV_ch48 (sessions 2-3)
    export_nt3, pine_nt3          NT3_PRV_ch48 (session 3): export JSON / golden, Pine generator (weights in libraries)
    linem_s4, linem_check         the EMA line M of PB_s64h24_d1 and its 'nocal' checks (session 4)

The indicator (PB_s64h24_d1, line M of session 5) does not import anything from here.
"""
