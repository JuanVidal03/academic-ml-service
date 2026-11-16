"""Dashboard aggregates."""

from fastapi import APIRouter

from dropout_service.dependencies import CurrentUserDep, SessionDep
from dropout_service.stats.schemas import StatsOverview
from dropout_service.stats.service import build_overview

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/overview", response_model=StatsOverview)
async def overview(session: SessionDep, _user: CurrentUserDep) -> StatsOverview:
    """Counts and shares for the dashboard.

    Any signed-in user: these are aggregates, not individual student records, so an adviser
    can see the shape of the cohort without being able to page the audit log.
    """
    return build_overview(session)
