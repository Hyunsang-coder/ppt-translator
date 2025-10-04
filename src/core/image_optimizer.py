"""Placeholder for future image optimisation routines."""

from __future__ import annotations

import logging

from pptx import Presentation

LOGGER = logging.getLogger(__name__)


class ImageOptimizer:
    """Optimise images in a presentation (currently a no-op placeholder)."""

    def optimise(self, presentation: Presentation) -> Presentation:
        """Return the presentation without modifying images.

        Args:
            presentation: Presentation object to (eventually) optimise.

        Returns:
            Unmodified presentation instance.
        """

        LOGGER.debug("Image optimisation placeholder invoked; no changes applied.")
        return presentation
