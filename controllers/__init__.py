"""HTTP controllers for independently generated course materials."""

from controllers.flashcards_controller import router as flashcards_router
from controllers.quiz_controller import router as quiz_router
from controllers.study_guide_controller import router as study_guide_router
from controllers.study_questions_controller import router as study_questions_router
from controllers.summary_controller import router as summary_router

routers = (
    summary_router,
    study_guide_router,
    flashcards_router,
    study_questions_router,
    quiz_router,
)
