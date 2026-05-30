"""src package — modules for live E. coli detection.

Modules:
    capture           USB / file / stream video source wrapper
    inference         YOLOv9 detector wrapper (Ultralytics backend)
    visualization     Bounding-box and HUD overlay rendering
    preprocessing     CLAHE and frame conditioning for microscopy
    config            YAML config loader → typed dataclasses
    logging_setup     Console + rotating-file logger setup
    main              CLI entry point and live detection loop
"""
