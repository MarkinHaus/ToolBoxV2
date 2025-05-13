import os
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import PyPDF2
import requests

from toolboxv2 import get_app
from toolboxv2.mods.TruthSeeker.arXivCrawler import (
    ArXivPDFProcessor,
    DocumentChunk,
    Insights,
    Paper,
    RobustPDFDownloader,
)
from toolboxv2.mods.TruthSeeker.module import byCode, codes, process, start, version
from toolboxv2.tests.a_util import async_test

default_test = get_app("TruthSeeker.Export").tb(mod_name="TruthSeeker", test_only=True, version=version)

class TestRobustPDFDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.downloader = RobustPDFDownloader(
            download_dir=self.temp_dir.name,
            log_file=os.path.join(self.temp_dir.name, 'test.log')
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('requests.Session.get')
    def test_download_pdf_success(self, mock_session):
        # Setup mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"fake pdf content"]
        mock_session.return_value.get.return_value.__enter__.return_value = mock_response

        # Test download
        url = "https://example.com/test.pdf"
        result = self.downloader.download_pdf(url)

        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith('.pdf'))

    @patch('requests.Session')
    def test_download_pdf_failure(self, mock_session):
        mock_session.return_value.get.side_effect = requests.exceptions.RequestException("Test error")

        with self.assertRaises(requests.exceptions.RequestException):
            self.downloader.download_pdf("https://example.com/test.pdf")

    def test_extract_text_from_pdf(self):
        # Create a test PDF file
        test_pdf_path = os.path.join(self.temp_dir.name, 'test.pdf')
        with open(test_pdf_path, 'wb') as f:
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_blank_page(width=72, height=72)
            pdf_writer.write(f)

        result = self.downloader.extract_text_from_pdf(test_pdf_path)
        self.assertIsInstance(result, list)

    def test_extract_images_from_pdf(self):
        # Create a test PDF with an image
        test_pdf_path = os.path.join(self.temp_dir.name, 'test_with_image.pdf')
        with open(test_pdf_path, 'wb') as f:
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_blank_page(width=72, height=72)
            pdf_writer.write(f)

        result = self.downloader.extract_images_from_pdf(test_pdf_path)
        self.assertIsInstance(result, list)

class TestArXivPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_tools = Mock()
        self.mock_tools.agent_memory.vector_store.semantic_similarity.return_value = 0.5

        self.processor = ArXivPDFProcessor(
            query="test query",
            tools=self.mock_tools,
            chunk_size=100,
            overlap=20,
            limiter=0.2,
            max_workers=2
        )

    def test_generate_queries(self):
        # Mock the tools.format_class response
        mock_queries = Mock()
        mock_queries.queries = ["query1", "query2"]
        self.mock_tools.format_class.return_value = mock_queries

        queries = self.processor.generate_queries()

        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)
        self.mock_tools.format_class.assert_called_once()

    def test_chunk_document(self):
        test_text = "This is a test document " * 50  # Create a longer text
        chunks = self.processor._chunk_document(test_text)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)

    @patch('arxiv.Search')
    def test_search_and_process_papers(self, mock_search):
        # Mock arxiv search results
        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.summary = "Test Summary"
        mock_result.pdf_url = "https://example.com/test.pdf"

        mock_search.return_value.results.return_value = [mock_result]

        # Mock PDF download and processing
        with patch.object(RobustPDFDownloader, 'download_pdf') as mock_download:
            with patch.object(RobustPDFDownloader, 'extract_text_from_pdf') as mock_extract:
                mock_download.return_value = "test.pdf"
                mock_extract.return_value = [{'page_number': 1, 'text': 'test content'}]

                papers = self.processor.search_and_process_papers(["test query"])

                self.assertIsInstance(papers, list)
                if papers:
                    self.assertIsInstance(papers[0], Paper)

    def test_generate_insights(self):
        # Create test papers
        test_papers = [
            Paper(
                title="Test Paper",
                summary="Test Summary",
                pdf_url="https://example.com/test.pdf",
                chunks=[
                    DocumentChunk(
                        content="Test content",
                        page_number=1,
                        relevance_score=0.8
                    )
                ],
                overall_relevance_score=0.8
            )
        ]

        # Mock tools responses
        mock_insights = Insights(
            is_true=True,
            summary="Test summary",
            key_point="Test key point"
        )
        self.mock_tools.config = {'agents-name-list':  ["InsightsAgent"]}
        self.mock_tools.format_class.return_value = mock_insights
        self.mock_tools.mini_task_completion.return_value = "Test summary"

        insights = self.processor.generate_insights(test_papers)

        self.assertIsInstance(insights, Insights)
        self.assertIsInstance(insights.summary, str)
        self.assertIsInstance(insights.key_point, str)

    def test_process_empty_results(self):
        # Test processing with no results
        with patch.object(ArXivPDFProcessor, 'generate_queries') as mock_generate:
            with patch.object(ArXivPDFProcessor, 'search_and_process_papers') as mock_search:
                mock_generate.return_value = ["test query"]
                mock_search.return_value = []

                papers, insights = self.processor.process()

                self.assertEqual(len(papers), 0)
                self.assertIsInstance(insights, Insights)
                self.assertIsNone(insights.is_true)

    @patch('arxiv.Search')
    def test_integration_process(self, mock_search):
        # Mock complete process
        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.summary = "Test Summary"
        mock_result.pdf_url = "https://example.com/test.pdf"



        mock_search.return_value.results.return_value = [mock_result]

        with patch.object(RobustPDFDownloader, 'download_pdf') as mock_download:
            with patch.object(RobustPDFDownloader, 'extract_text_from_pdf') as mock_extract:
                with patch.object(self.mock_tools, 'format_class') as mock_format_class:
                    mock_download.return_value = "test.pdf"
                    mock_extract.return_value = [{'page_number': 1, 'text': 'test content'}]
                    def x():
                        return None
                    x.queries = [""]
                    mock_format_class.return_value = x

                    papers, insights = self.processor.process()

                    self.assertIsInstance(papers, list)
                    self.assertIsInstance(insights, Insights)

class TestTruthSeeker(unittest.TestCase):
    def setUp(self):
        # Mock the App class
        self.mock_app = Mock()
        self.mock_app.get_mod.return_value = Mock()

        # Setup mock for run_any that returns iterable dict
        self.mock_app.run_any.return_value = {
            "1": {"name": "template1"},
            "2": {"name": "template2"}
        }

        # Mock RequestSession
        self.mock_request = Mock()
        self.mock_request.json = AsyncMock()

    @patch('os.path.join')
    @patch('builtins.open', create=True)
    def test_start_initialization(self, mock_open, mock_join):
        """Test the start function initializes correctly"""
        # Setup mock file handling
        mock_file = Mock()
        mock_file.read.return_value = "test content"
        mock_open.return_value.__enter__.return_value = mock_file

        # Call start function
        start(self.mock_app)

        # Verify app initialization calls
        self.mock_app.get_mod.assert_called_with("CodeVerification")
        self.mock_app.run_any.assert_any_call(("CodeVerification", "init_scope"), scope="TruthSeeker")
        self.mock_app.run_any.assert_any_call(("CodeVerification", "init_scope"), scope="TruthSeeker-promo")

    @async_test
    async def test_codes_valid_request(self):
        """Test the codes function with valid input"""
        # Mock request data
        test_data = {
            "query": "test query",
            "depth": "Q",
            "promoCode": "PROMO15",
            "ontimeCode": "TEST123"
        }
        self.mock_request.json.return_value = test_data

        # Mock code verification
        self.mock_app.run_any.return_value = {
            "template_name": "Promo15",
            "usage_type": "one_time"
        }

        result = await codes(self.mock_app, self.mock_request)

        self.assertTrue(result['valid'])
        self.assertIn('ontimeKey', result)
        self.assertIn('ppc', result)

    @async_test
    async def test_codes_invalid_promo(self):
        """Test the codes function with invalid promo code"""
        test_data = {
            "query": "test query",
            "depth": "I",
            "promoCode": "INVALID",
            "ontimeCode": "TEST123"
        }
        self.mock_request.json.return_value = test_data

        # Mock invalid promo code verification
        self.mock_app.run_any.return_value = None

        result = await codes(self.mock_app, self.mock_request)

        self.assertIn('ppc', result)
        self.assertTrue(result['ppc']['price'] > 0)

    @async_test
    async def test_process_valid_request(self):
        """Test the process function with valid input"""
        test_data = {
            "query": "test query",
            "depth": "Q",
            "ontimeKey": "VALID_KEY",
            "email": "test@example.com"
        }
        self.mock_request.json.return_value = test_data

        # Mock valid key verification
        self.mock_app.run_any.return_value = {
            "template_name": "PROCESS",
            "usage_type": "timed",
            "uses_count": 1
        }

        # Mock ArXivPDFProcessor
        with patch('toolboxv2.mods.TruthSeeker.module.ArXivPDFProcessor') as mock_processor:
            mock_insights = MagicMock()
            mock_insights.is_true = "True"
            mock_insights.summary = "Test summary"
            mock_insights.key_point = "Point1>\n\n<Point2"

            mock_processor.return_value.process.return_value = ([], mock_insights)

            result = await process(self.mock_app, self.mock_request)

            self.assertEqual(result['is_true'], "True")
            self.assertEqual(result['summary'], "Test summary")

    @async_test
    async def test_process_invalid_key(self):
        """Test the process function with invalid key"""
        test_data = {
            "query": "test query",
            "depth": "Q",
            "ontimeKey": "INVALID_KEY",
            "email": "test@example.com"
        }
        self.mock_request.json.return_value = test_data

        # Mock invalid key verification
        self.mock_app.run_any.return_value = None

        result = await process(self.mock_app, self.mock_request)

        self.assertEqual(result['summary'], "INVALID QUERY")
        self.assertEqual(result['insights'], [])
        self.assertEqual(result['papers'], [])

    def test_byCode_functionality(self):
        """Test the byCode function"""
        test_request = Mock()
        test_request.json.return_value = ["payKey", "codeClass", "ontimeKey"]

        result = byCode(self.mock_app, test_request)

        self.assertEqual(result, {'code': 'code'})

def run_test_suite(test_class=None, test_name=None, verbosity=2):
    """
    Run specific test class or test case.

    Args:
        test_class: The test class to run (optional)
        test_name: Specific test method name to run (optional)
        verbosity: Output detail level (default=2)

    Returns:
        TestResult object
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if test_class and test_name:
        # Run specific test method
        suite.addTest(test_class(test_name))
    elif test_class:
        # Run all tests in the class
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    else:
        # Run all tests
        suite.addTests(loader.loadTestsFromModule(sys.modules[__name__]))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def run_pdf_downloader_tests(test_name=None):
    """Run TestRobustPDFDownloader tests"""
    return run_test_suite(TestRobustPDFDownloader, test_name)


def run_arxiv_processor_tests(test_name=None):
    """Run TestArXivPDFProcessor tests"""
    return run_test_suite(TestArXivPDFProcessor, test_name)

def run_truth_seeker_tests(test_name=None):
    """Run TestTruthSeeker tests"""
    return run_test_suite(TestTruthSeeker, test_name)


def run_specific_test(test_class, test_name):
    """Run a specific test from a test class"""
    return run_test_suite(test_class, test_name)

@default_test
def run_all_tests():
    """Run all test classes"""
    return run_test_suite()

@default_test
def test_pdf_download():
    """Run only PDF download tests"""
    return run_specific_test(
        TestRobustPDFDownloader,
        'test_download_pdf_success'
    )

@default_test
def test_truth_seeker():
    """Run only PDF download tests"""
    return run_specific_test(
        TestTruthSeeker,
        'test_truth_seeker_success'
    )

@default_test
def test_arxiv_search():
    """Run only ArXiv search tests"""
    return run_specific_test(
        TestArXivPDFProcessor,
        'test_search_and_process_papers'
    )


@default_test
@async_test
async def test_web_interactions():
    from toolboxv2.tests.test_web import run_in_valid_session_tests

    def base_test(_): return [
        {'type': 'goto', 'url': 'localhost:5000/TruthSeeker'},
        {'type': 'sleep', 'time': 2},
        {'type': 'test', 'selector': '.logo h1'},
        {'type': 'test', 'selector': 'nav a'},
        {'type': 'test', 'selector': '#query-input'},
        {'type': 'test', 'selector': '#research-depth'},
        {'type': 'test', 'selector': '#status-display'}
    ]

    def search_test(_): return [
        {'type': 'goto', 'url': 'localhost:5000/TruthSeeker'},
        {'type': 'type', 'selector': '#query-input', 'text': 'Test research query'},
        {'type': 'select', 'selector': '#research-depth', 'value': 'indeep'},
        {'type': 'click', 'selector': '#search-form button[type="submit"]'},
        {'type': 'sleep', 'time': 2},
        {'type': 'test', 'selector': '#ppc-section'},
        {'type': 'test', 'selector': '#results-container'}
    ]

    def promo_code_test(_): return [
        {'type': 'goto', 'url': 'localhost:5000/TruthSeeker'},
        {'type': 'type', 'selector': '#promo-code', 'text': 'TESTCODE'},
        {'type': 'click', 'selector': '#apply-promo-button'},
        {'type': 'type', 'selector': '#ontime-code', 'text': 'TESTTIME'},
        {'type': 'click', 'selector': '#apply-ontime-button'},
        {'type': 'sleep', 'time': 1}
    ]

    def modal_tests(_): return [
        {'type': 'goto', 'url': 'localhost:5000/TruthSeeker'},
        {'type': 'click', 'selector': 'button[data-modal-target="payment-dialog"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#payment-dialog[open]'},
        {'type': 'click', 'selector': '#payment-dialog .close-dialog'},
        {'type': 'click', 'selector': 'button[data-modal-target="terms-dialog"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#terms-dialog[open]'},
        {'type': 'click', 'selector': '#terms-dialog .close-dialog'},
        {'type': 'click', 'selector': 'button[data-modal-target="privacy-dialog"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#privacy-dialog[open]'},
        {'type': 'click', 'selector': '#privacy-dialog .close-dialog'},
        {'type': 'click', 'selector': 'button[data-modal-target="imprint-dialog"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#imprint-dialog[open]'},
        {'type': 'click', 'selector': '#imprint-dialog .close-dialog'}
    ]

    def navigation_test(_): return [
        {'type': 'goto', 'url': 'localhost:5000/TruthSeeker'},
        {'type': 'click', 'selector': 'nav a[href="#pricing"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#pricing'},
        {'type': 'click', 'selector': 'nav a[href="#about"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#about'},
        {'type': 'click', 'selector': 'nav a[href="#home"]'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#home'}
    ]

    def email_test(_): return [
        {'type': 'goto', 'url': 'localhost:5000/TruthSeeker'},
        {'type': 'type', 'selector': '#Email', 'text': 'test@example.com'},
        {'type': 'click', 'selector': '#add-email-button'},
        {'type': 'sleep', 'time': 1},
        {'type': 'test', 'selector': '#EmailS'}
    ]

    res = await run_in_valid_session_tests(
        [base_test, search_test,promo_code_test,modal_tests,navigation_test], headless=False)
    print(res)


if __name__ == '__main__':
    unittest.main()

