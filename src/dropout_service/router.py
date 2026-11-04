"""Assembles the versioned API surface.

The ``/api/v1`` prefix is declared here and nowhere else, so introducing a ``v2`` later is
a second ``include_router`` rather than a search-and-replace across every slice.
"""

from fastapi import APIRouter

from dropout_service.predictions.router import router as predictions_router
from dropout_service.system.router import meta_router

API_V1_PREFIX = "/api/v1"

api_router = APIRouter(prefix=API_V1_PREFIX)
api_router.include_router(predictions_router)
api_router.include_router(meta_router)

# When auth arrives, register it here:
#     from dropout_service.auth.router import router as auth_router
#     api_router.include_router(auth_router)
