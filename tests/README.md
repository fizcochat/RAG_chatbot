# RAG Chatbot Testing Framework

This directory contains a sophisticated test suite for the Fiscozen RAG (Retrieval-Augmented Generation) chatbot. The tests are designed to validate various aspects of the chatbot's functionality, from individual components to end-to-end performance.

## Test Structure

The test suite is organized into several modules, each with a specific focus:

- **Basic Tests** (`test_basic.py`): Simple tests for basic functionality
- **Retrieval Tests** (`test_retrieval.py`): Tests for the vector search and retrieval capabilities
- **Advanced Tests** (`test_advanced.py`): Modularized tests for RAG components, conversation flow, edge cases, and security
- **Response Quality Tests** (`test_response_quality.py`): Tests that evaluate the quality and relevance of responses
- **Integration Tests** (`test_integration.py`): End-to-end tests and simulated user sessions
- **Configuration** (`conftest.py`): Shared fixtures, test hooks, and reporting utilities

## Running the Tests

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
pytest
pytest-xdist     # For parallel test execution
nltk             # For NLP-based tests
```

### Basic Test Execution

Run all tests:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_advanced.py
```

Run a specific test:

```bash
pytest tests/test_advanced.py::TestRagComponents::test_query_refinement
```

### Test Categories

You can run tests by category using markers:

```bash
pytest -m retrieval    # Run retrieval-related tests only
pytest -m quality      # Run response quality tests
pytest -m integration  # Run integration tests
pytest -m performance  # Run performance tests
```

### Performance Optimization

Run tests in parallel (useful for large test suites):

```bash
pytest -xvs -n auto
```

### Integration Tests (with real API calls)

By default, integration tests that make actual API calls are skipped. To run these tests:

```bash
RUN_INTEGRATION_TESTS=1 pytest -m integration
```

Make sure your environment variables for API keys are properly set when running integration tests.

## Test Reports

When tests are run, a performance report is automatically generated in the `test_reports` directory. These reports include:

- Test summary (pass/fail counts)
- Performance metrics for each test
- Detailed results for failed tests

To view the latest report summary after running tests:

```bash
cat test_reports/$(ls -t test_reports | head -1)
```

## Adding New Tests

When adding new tests:

1. Place them in the appropriate test file based on what they're testing
2. Use the existing fixtures in `conftest.py` where possible
3. Add appropriate markers for test categorization
4. Ensure tests are isolated (don't depend on the state of other tests)
5. Follow the modular test structure to keep tests maintainable

## Mock vs. Real Testing

The test suite supports both mock-based testing (for speed and reliability) and real API testing (for integration validation):

- Most tests use mocks to avoid API costs and ensure consistent behavior
- Integration tests can use actual API calls when needed
- Configure the behavior in the test fixtures or with environment variables

## Troubleshooting

If tests are failing with API errors:
- Check that API keys are properly set in environment variables
- Verify Pinecone index exists and is properly configured
- Check OpenAI API rate limits

If tests are slow:
- Use mock fixtures instead of real API calls
- Run tests in parallel with `-n auto`
- Consider using test categories to run only relevant tests

## Contributing

When contributing new tests:
1. Follow the existing structure and naming conventions
2. Add appropriate docstrings explaining what the test is verifying
3. Use fixtures to minimize repeated setup code
4. Consider both positive and negative test cases 