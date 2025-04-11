# FastText module initialization
"""
This makes the fasttext directory a proper Python package,
allowing imports like 'from fast_text.relevance import FastTextRelevanceChecker'
"""

# Import the FastTextRelevanceChecker to make it available directly
try:
    from .relevance import FastTextRelevanceChecker
except ImportError:
    # Provide a message but don't fail - allows the package to be imported even if relevance.py isn't there yet
    import warnings
    warnings.warn("Could not import FastTextRelevanceChecker. Make sure relevance.py exists in the fast_text directory.")

"""
FastText module for tax-related text classification.
"""

__all__ = ['FastTextRelevanceChecker'] 