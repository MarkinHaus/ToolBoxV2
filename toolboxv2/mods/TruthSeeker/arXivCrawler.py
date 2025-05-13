"""
ArXiv Crawler for TruthSeeker.
Main module for processing research queries.
"""
import asyncio
import logging
import os
import threading
import time
import uuid

from pydantic import BaseModel, Field
from urllib3 import Retry

from toolboxv2 import get_app
from toolboxv2.mods.isaa.base.KnowledgeBase import TextSplitter

from .pdf_processor import RobustPDFDownloader

# Import the new modular components
from .research_processor import ResearchProcessor
from .sources.arxiv_source import Paper, search_papers
from .text_splitter import TextSplitter


class RobustPDFDownloader:
    def __init__(self, max_retries=5, backoff_factor=0.3,
                 download_dir='downloads',
                 log_file='pdf_downloader.log'):
        """
        Initialize the robust PDF downloader with configurable retry mechanisms

        Args:
            max_retries (int): Maximum number of download retries
            backoff_factor (float): Exponential backoff multiplier
            download_dir (str): Directory to save downloaded files
            log_file (str): Path for logging download activities
        """
        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create download directories
        self.download_dir = download_dir
        self.pdf_dir = os.path.join(download_dir, 'pdfs')
        self.images_dir = os.path.join(download_dir, 'images')
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Configure retry strategy
        self.retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor
        )
        self.adapter = HTTPAdapter(max_retries=self.retry_strategy)

        # Create session with retry mechanism
        self.session = requests.Session()
        self.session.mount("https://", self.adapter)
        self.session.mount("http://", self.adapter)

    def download_pdf(self, url, filename=None):
        """
        Download PDF with robust retry mechanism

        Args:
            url (str): URL of the PDF
            filename (str, optional): Custom filename for PDF

        Returns:
            str: Path to downloaded PDF file
        """
        try:
            # Generate filename if not provided
            if not filename:
                filename = url.split("/")[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'

            file_path = os.path.join(self.pdf_dir, filename)

            # Attempt download with timeout and stream
            with self.session.get(url, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            self.logger.info(f"Successfully downloaded: {file_path}")
            return file_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Download failed for {url}: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from each page of a PDF

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            list: Text content from each page
        """
        try:
            page_texts = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    page_texts.append({
                        'page_number': page_num,
                        'text': text
                    })

            return page_texts

        except Exception as e:
            self.logger.error(f"Text extraction failed for {pdf_path}: {e}")
            return []

    def extract_images_from_pdf(self, pdf_path):
        """
        Extract images from PDF and save them

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            list: Paths of extracted images
        """
        extracted_images = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        for img_index, image in enumerate(page.images):
                            img_data = image.data
                            img = Image.open(io.BytesIO(img_data))

                            img_filename = f'page_{page_num}_img_{img_index}.png'
                            img_path = os.path.join(self.images_dir, img_filename)

                            img.save(img_path)
                            extracted_images.append(img_path)
                    except Exception as inner_e:
                        self.logger.warning(f"Image extraction issue on page {page_num}: {inner_e}")

            return extracted_images

        except Exception as e:
            self.logger.error(f"Image extraction failed for {pdf_path}: {e}")
            return []

class Insights(BaseModel):
    is_true: bool | None = Field(..., description="if the Statement in the query is True or not basd on the papers")
    summary: str = Field(..., description="Comprehensive summary addressing the query")
    key_point: str | None = Field(..., description="Most important findings")

class ISTRUE(BaseModel):
    value: bool | None = Field(..., description="if the Statement in the query is True or not basd on the papers")

class DocumentChunk(BaseModel):
    content: str
    page_number: int
    relevance_score: float = 0.0

class Paper(BaseModel):
    title: str
    summary: str
    pdf_url: str
    ref_pages: list[int] | None = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    overall_relevance_score: float = 0.0

class RelevanceAssessment(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    key_sections: list[str] = Field(default_factory=list)

def search_papers(query: str, max_results=10) -> list[Paper]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for result in arxiv.Client().results(search):
        paper = Paper(
            title=result.title,
            summary=result.summary,
            pdf_url=result.pdf_url
        )
        results.append(paper)
    return results



# Import the new modular components

class ArXivPDFProcessor:
    """
    Main processor for research queries.
    This is a wrapper around the new ResearchProcessor for backward compatibility.
    """
    def __init__(self,
                 query: str,
                 tools,
                 chunk_size: int = 1_000_000,
                 overlap: int = 2_000,
                 max_workers=None,
                 num_search_result_per_query=6,
                 max_search=6,
                 download_dir="pdfs",
                 callback=None,
                 num_workers=None):
        """Initialize the ArXiv PDF processor.

        Args:
            query: Research query
            tools: Tools module
            chunk_size: Size of text chunks for processing
            overlap: Overlap between chunks
            max_workers: Maximum number of worker threads
            num_search_result_per_query: Number of search results per query
            max_search: Maximum number of search queries
            download_dir: Directory to save downloaded files
            callback: Callback function for status updates
            num_workers: Number of worker threads
        """
        # Create the new research processor
        self.processor = ResearchProcessor(
            query=query,
            tools=tools,
            chunk_size=chunk_size,
            overlap=overlap,
            max_workers=max_workers,
            num_search_result_per_query=num_search_result_per_query,
            max_search=max_search,
            download_dir=download_dir,
            callback=callback,
            num_workers=num_workers
        )

        # Copy attributes for backward compatibility
        self.insights_generated = False
        self.queries_generated = False
        self.query = query
        self.tools = tools
        self.mem = tools.get_memory()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers
        self.nsrpq = num_search_result_per_query
        self.max_search = max_search
        self.download_dir = download_dir
        self.parser = RobustPDFDownloader(download_dir=download_dir)
        self.callback = callback if callback is not None else lambda status: None
        self.mem_name = None
        self.current_session = None
        self.all_ref_papers = 0
        self.last_insights_list = None
        self.all_texts_len = 0
        self.f_texts_len = 0
        self.s_id = str(uuid.uuid4())
        self.semantic_model = self.processor.semantic_model
        self._query_progress = {}
        self._progress_lock = threading.Lock()
        self.num_workers = self.processor.num_workers

    def _update_global_progress(self) -> float:
        """Calculate overall progress considering all processing phases."""
        return self.processor._update_global_progress()

    async def search_and_process_papers(self, queries: list[str]) -> list[Paper]:
        """Search for and process papers based on queries.

        Args:
            queries: List of search queries

        Returns:
            List of processed papers
        """
        # Use the new processor to search and process papers
        unified_papers = await self.processor.search_and_process_papers(queries)

        # Convert UnifiedPaper objects to Paper objects for backward compatibility
        papers = []
        for paper in unified_papers:
            if paper.source == "arxiv":
                # Convert to the old Paper format
                arxiv_paper = Paper(
                    title=paper.title,
                    authors=paper.authors,
                    summary=paper.summary,
                    url=paper.url,
                    pdf_url=paper.pdf_url,
                    published=paper.published,
                    updated=paper.source_specific_data.get("updated", ""),
                    categories=paper.source_specific_data.get("categories", []),
                    paper_id=paper.paper_id
                )
                papers.append(arxiv_paper)

        # Update attributes for backward compatibility
        self.all_ref_papers = self.processor.all_ref_papers
        self.all_texts_len = self.processor.all_texts_len
        self.f_texts_len = self.processor.f_texts_len

        return papers

    def send_status(self, step: str, progress: float = None, additional_info: str = ""):
        """Send status update via callback."""
        if progress is None:
            progress = self._update_global_progress()
        self.callback({
            "step": step,
            "progress": progress,
            "info": additional_info
        })

    def generate_queries(self) -> list[str]:
        self.send_status("Generating search queries")
        self.queries_generated = False

        class ArXivQueries(BaseModel):
            queries: list[str] = Field(..., description="List of ArXiv search queries (en)")

        try:
            query_generator: ArXivQueries = self.tools.format_class(
                ArXivQueries,
                f"Generate a list of precise ArXiv search queries to comprehensively address: {self.query}"
            )
            queries = [self.query] + query_generator["queries"]
        except Exception:
            self.send_status("Error generating queries", additional_info="Using default query.")
            queries = [self.query]

        if len(queries[:self.max_search]) > 0:
            self.queries_generated = True
        return queries[:self.max_search]

    def init_process_papers(self):
        self.mem.create_memory(self.mem_name, model_config={"model_name": "anthropic/claude-3-5-haiku-20241022"})
        self.send_status("Memory initialized")


    async def generate_insights(self, queries) -> dict:
        self.send_status("Generating insights")
        query = self.query
        # max_it = 0
        results = await self.mem.query(query=query, memory_names=self.mem_name, unified_retrieve=True, query_params={
            "max_sentences": 25})
        #query = queries[min(len(queries)-1, max_it)]

        self.insights_generated = True
        self.send_status("Insights generated", progress=1.0)
        return results

    async def extra_query(self, query, query_params=None, unified_retrieve=True):
        self.send_status("Processing follow-up query", progress=0.5)
        results = await self.mem.query(query=query, memory_names=self.mem_name,
                                                      query_params=query_params, unified_retrieve=unified_retrieve)
        self.send_status("Processing follow-up query Done", progress=1)
        return results

    def generate_mem_name(self):
        class UniqueMemoryName(BaseModel):
            """unique memory name based on the user query"""
            name: str
        return self.tools.get_agent("thinkm").format_class(UniqueMemoryName, self.query).get('name', '_'.join(self.query.split(" ")[:3]))

    def initialize(self, session_id, second=False):
        self.current_session = session_id
        self.insights_generated = False
        self.queries_generated = False
        if second:
            return
        self.mem_name = self.generate_mem_name().strip().replace("\n", '') + '_' + session_id
        self.init_process_papers()

    async def process(self, query=None) -> tuple[list[Paper], dict]:
        if query is not None:
            self.query = query
        self.send_status("Starting research process")
        t0 = time.perf_counter()
        self.initialize(self.s_id, query is not None)

        queries = self.generate_queries()

        papers = await self.search_and_process_papers(queries)

        if len(papers) == 0:
            class UserQuery(BaseModel):
                """Fix all typos and clear the original user query"""
                new_query: str
            self.query= self.tools.format_class(
                UserQuery,
                self.query
            )["new_query"]
            queries = self.generate_queries()
            papers = await self.search_and_process_papers(queries)

        insights = await self.generate_insights(queries)

        elapsed_time = time.perf_counter() - t0
        self.send_status("Process complete", progress=1.0,
                         additional_info=f"Total time: {elapsed_time:.2f}s, Papers analyzed: {len(papers)}/{self.all_ref_papers}")

        return papers, insights

    @staticmethod
    def estimate_processing_metrics(query_length: int, **config) -> (float, float):
        """Return estimated time (seconds) and price for processing."""
        total_papers = config['max_search'] * config['num_search_result_per_query']
        median_text_length = 100000  # 10 pages * 10000 characters

        # Estimated chunks to process
        total_chunks = total_papers * (median_text_length / config['chunk_size']) + 1 / config['overlap']
        processed_chunks = total_chunks * 0.45
        total_chars = TextSplitter(config['chunk_size'],
                     config['overlap']
                     ).approximate(config['chunk_size'] * processed_chunks)
        # Time estimation (seconds)
        .75 / config['chunk_size']  # Hypothetical time per chunk in seconds
        w = (config.get('num_workers', 16) if config.get('num_workers', 16) is not None else 16 / 10)
        # Processing_ time - Insights Genration - Insights Query   -   Indexing Time     -    Download Time     -       workers   -   Query Genration time - Ui - Init Db
        estimated_time = ((8+total_papers*0.012)+(total_chunks/20000) * .005 + (total_chunks/2) * .0003 + total_papers * 2.8 ) / w + (0.25 * config['max_search']) + 6 + 4

        price_per_char = 0.0000012525
        price_per_t_chunk =  total_chars * price_per_char
        estimated_price = price_per_t_chunk ** 1.7

        # estimated_price = 0 if query_length < 420 and estimated_price < 5 else estimated_price
        if estimated_time < 10:
            estimated_time = 10
        if estimated_price < .04:
            estimated_price = .04
        return round(estimated_time, 2), round(estimated_price, 4)

async def main(query: str = "Beste strategien in bretspielen sitler von katar"):
    """Main execution function"""
    with Spinner("Init Isaa"):
        tools = get_app("ArXivPDFProcessor", name=None).get_mod("isaa")
        tools.init_isaa(build=True)
    processor = ArXivPDFProcessor(query, tools=tools)
    papers, insights = await processor.process()

    print("Generated Insights:", insights)
    print("Generated Insights_list:", processor.last_insights_list)
    kb = tools.get_memory(processor.mem_name)
    print(await kb.query_concepts("AI"))
    print(await kb.retrieve("Evaluation metrics for assessing AI Agent performance"))
    print(kb.concept_extractor.concept_graph.concepts.keys())
    kb.vis(output_file="insights_graph.html")
    kb.save("mem.plk")
    # await get_app("ArXivPDFProcessor", name=None).a_idle()
    return insights


if __name__ == "__main__":
    asyncio.run(main("Beste strategien AI Agents Development"))
