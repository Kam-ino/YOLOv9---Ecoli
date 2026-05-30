"""FastAPI backend for live E. coli detection.

Wraps the standalone inference modules in :mod:`src` with HTTP and
MJPEG streaming endpoints, and (in production) serves the built React
frontend as static files. Designed to run on a single port so the same
binary can be deployed on a Raspberry Pi without a separate web server.
"""
