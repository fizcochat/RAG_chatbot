import pytest
import os
import json
from unittest.mock import MagicMock
import time
from collections import defaultdict
import streamlit as st

# ==================== TEST REPORTING ====================

class TestMetricsCollector:
    """Helper class to collect test metrics and generate reports"""
    def __init__(self):
        self.test_times = defaultdict(list)
        self.test_results = {}
        
    def record_test_time(self, test_name, execution_time):
        self.test_times[test_name].append(execution_time)
        
    def record_test_result(self, test_name, passed, details=None):
        self.test_results[test_name] = {
            'passed': passed,
            'details': details or {}
        }
        
    def generate_report(self):
        """Generate a performance report of all tests"""
        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results.values() if r['passed']),
                'failed_tests': sum(1 for r in self.test_results.values() if not r['passed'])
            },
            'performance': {
                category: {
                    'avg_time': sum(times) / len(times) if times else 0,
                    'min_time': min(times) if times else 0,
                    'max_time': max(times) if times else 0,
                    'total_time': sum(times) if times else 0
                } 
                for category, times in self.test_times.items()
            },
            'results': self.test_results
        }
        return report


@pytest.fixture(scope="session")
def metrics_collector():
    """Fixture to provide a metrics collector for the test session"""
    collector = TestMetricsCollector()
    yield collector
    
    # At the end of session, generate and save report
    report = collector.generate_report()
    
    # Create reports directory if it doesn't exist
    os.makedirs('test_reports', exist_ok=True)
    
    # Save report to JSON file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f'test_reports/test_report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary to console
    print("\n=== Test Performance Report ===")
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    
    if report['performance']:
        print("\nPerformance by category:")
        for category, perf in report['performance'].items():
            print(f"  {category}: avg={perf['avg_time']:.3f}s, min={perf['min_time']:.3f}s, max={perf['max_time']:.3f}s")

# ==================== TEST HOOKS ====================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    
    # Get the metrics collector if available
    metrics_collector = item.config.pluginmanager.get_plugin("metrics_collector")
    if not metrics_collector and hasattr(item, "funcargs") and "metrics_collector" in item.funcargs:
        metrics_collector = item.funcargs["metrics_collector"]
    
    if metrics_collector and rep.when == "call":
        test_name = item.nodeid
        
        # Record test result
        metrics_collector.record_test_result(
            test_name, 
            rep.passed, 
            {"outcome": rep.outcome, "longrepr": str(rep.longrepr) if rep.longrepr else None}
        )
        
        # Record execution time if available
        if hasattr(rep, "duration"):
            metrics_collector.record_test_time(test_name, rep.duration)

# ==================== SHARED FIXTURES ====================

@pytest.fixture
def mock_env_vars():
    """Fixture to set mock environment variables for testing"""
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_pinecone_key = os.environ.get("PINECONE_API_KEY")
    
    # Set mock keys for testing
    os.environ["OPENAI_API_KEY"] = "mock-openai-key"
    os.environ["PINECONE_API_KEY"] = "mock-pinecone-key"
    
    yield
    
    # Restore original environment
    if original_openai_key:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    else:
        del os.environ["OPENAI_API_KEY"]
        
    if original_pinecone_key:
        os.environ["PINECONE_API_KEY"] = original_pinecone_key
    else:
        del os.environ["PINECONE_API_KEY"]

@pytest.fixture
def sample_vectorstore_results():
    """Fixture to provide sample vectorstore results for testing"""
    return [
        MagicMock(
            page_content="The standard IVA (VAT) rate in Italy is 22%, with reduced rates of 10% and 4% for specific goods and services.",
            metadata={"source": "tax_guide.pdf", "page": 12}
        ),
        MagicMock(
            page_content="VAT registration is required for businesses with annual turnover exceeding €65,000.",
            metadata={"source": "business_requirements.pdf", "page": 45}
        ),
        MagicMock(
            page_content="The 'forfettario' regime allows small businesses with revenue below thresholds to pay a substitute tax instead of regular IVA.",
            metadata={"source": "tax_regimes.pdf", "page": 8}
        ),
    ]

@pytest.fixture
def reset_streamlit_session():
    """Fixture to reset Streamlit session state between tests"""
    # Backup current session state
    backup = {}
    if 'responses' in st.session_state:
        backup['responses'] = st.session_state['responses'].copy()
    if 'requests' in st.session_state:
        backup['requests'] = st.session_state['requests'].copy()
    
    # Set default session state
    st.session_state['responses'] = ["How can I assist you?"]
    st.session_state['requests'] = []
    
    yield
    
    # Restore session state
    if 'responses' in backup:
        st.session_state['responses'] = backup['responses']
    if 'requests' in backup:
        st.session_state['requests'] = backup['requests']

# Add pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "retrieval: tests for retrieval component")
    config.addinivalue_line("markers", "integration: tests requiring real API calls")
    config.addinivalue_line("markers", "performance: tests measuring system performance")
    config.addinivalue_line("markers", "quality: tests assessing response quality") 