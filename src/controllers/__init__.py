"""HTTP controllers for independently generated course materials."""

from src.controllers.flashcards_controller import router as flashcards_router
from src.controllers.podcast_controller import router as podcast_router
from src.controllers.quiz_controller import router as quiz_router
from src.controllers.study_guide_controller import router as study_guide_router
from src.controllers.study_questions_controller import router as study_questions_router
from src.controllers.summary_controller import router as summary_router

routers = (
    summary_router,
    study_guide_router,
    flashcards_router,
    study_questions_router,
    quiz_router,
    podcast_router,
)
