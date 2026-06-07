"""Legacy DS1↔DS2 linking — not used in university pipeline scope.

DS3 enriches DS2 via match_key in bookrec/features/content.py.
"""

from bookrec.resolution.book_linker import link_books_to_catalog, link_external_books

__all__ = ["link_books_to_catalog", "link_external_books"]
