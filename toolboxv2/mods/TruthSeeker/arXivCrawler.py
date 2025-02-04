import threading
import time
import uuid
from typing import List, Tuple, Optional, Dict

import asyncio
from pydantic import BaseModel, Field
from toolboxv2 import get_app, Spinner
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import arxiv
import os
import requests
from requests.adapters import HTTPAdapter, Retry
import PyPDF2
from PIL import Image
import io
import time
import logging

from urllib3 import Retry

from toolboxv2.mods.isaa.base.AgentUtils import AISemanticMemory


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
    is_true: Optional[bool] = Field(..., description="if the Statement in the query is True or not basd on the papers")
    summary: str = Field(..., description="Comprehensive summary addressing the query")
    key_point: Optional[str] = Field(..., description="Most important findings")

class ISTRUE(BaseModel):
    value: Optional[bool] = Field(..., description="if the Statement in the query is True or not basd on the papers")

class DocumentChunk(BaseModel):
    content: str
    page_number: int
    relevance_score: float = 0.0

class Paper(BaseModel):
    title: str
    summary: str
    pdf_url: str
    ref_pages: Optional[List[int]] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    overall_relevance_score: float = 0.0

class RelevanceAssessment(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    key_sections: List[str] = Field(default_factory=list)

def search_papers(query: str, max_results=10) -> List[Paper]:
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


class ArXivPDFProcessor:
    def __init__(self,
                 query: str,
                 tools,
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 limiter=0.2,
                 max_workers=None,
                 num_search_result_per_query=3,
                 max_search=6,
                 download_dir="pdfs",
                 callback=None):
        # Initialize components
        self.mem_name = None
        self.current_session = None
        self.last_insights_list = None
        self.max_workers = max_workers
        self.tools = tools
            #with Spinner("Building agent"):
            #    self.tools.init_isaa(build=True)

        # chunking parameters
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.nsrpq = num_search_result_per_query
        self.callback = callback if callback is not None else lambda x:None
        self.max_search = max_search

        # query
        self.query = query
        self.limiter = limiter

        self.temp = True
        self.parser = RobustPDFDownloader(download_dir=download_dir)

        # Ref Papers
        self.all_ref_papers = 0
        self.initialize(str(uuid.uuid4()))

    def generate_queries(self) -> List[str]:
        """Generate search queries based on the query"""

        class ArXivQueries(BaseModel):
            queries: List[str] = Field(..., description="List of ArXiv search queries (en)")

        try:
            query_generator: ArXivQueries = self.tools.format_class(
                ArXivQueries,
                f"Generate a list of precise ArXiv search queries to comprehensively address: {self.query}"
            )
        except Exception:
            self.nsrpq = self.nsrpq * self.max_search
            return [self.query]

        query_generator = [self.query] + query_generator["queries"]
        return query_generator[:self.max_search]

    def init_process_papers(self):
        self.tools.get_memory().create_memory(self.mem_name)
        self.callback({})


    async def search_and_process_papers(self, queries: List[str]) -> List[Paper]:
        """Search ArXiv, download PDFs, and process them in parallel batches."""
        all_papers = []
        lock = threading.Lock()

        async def process_query(query):
            papers = search_papers(query, max_results=self.nsrpq)
            with lock:
                self.all_ref_papers += len(papers)

            for paper in papers:
                try:
                    # Download PDF
                    pdf_path = self.parser.download_pdf(paper.pdf_url, 'temp_pdf' + paper.title[:20])
                    pdf_pages = self.parser.extract_text_from_pdf(pdf_path)
                    if self.temp:
                        os.remove(pdf_path)
                except Exception as e:
                    print(f"Error processing PDF {paper.pdf_url}: {e}")
                    continue

                #try:
                print("Adding ", len(pdf_pages), " chunks to document ", self.mem_name)
                await self.tools.get_memory().add_data(memory_name=self.mem_name, data=[page['text'] for page in pdf_pages])
                #except ValueError as e:
                #    print(f"Error processing PDF {paper.pdf_url}: {e}")
                #    continue
            return papers

        for query_ in tqdm(queries, total=len(queries), desc="Processing queries"):
            result = await process_query(query_)
            all_papers.append(result)
        # Use ThreadPoolExecutor for parallel processing
        # with ThreadPoolExecutor() as executor:
        #     future_to_query = {executor.submit(process_query, query): query for query in queries}

        #     for future in tqdm(as_completed(future_to_query), total=len(queries), desc="Processing queries"):
        #         query = future_to_query[future]
        #         try:
        #             result = future.result()
        #             all_papers.append(result)
        #         except Exception as e:
        #             print(f"Error processing query '{query}': {e}")
        return all_papers

    async def generate_insights(self) -> dict:
        """Generate insights using format_class"""
        results = await self.tools.get_memory().directly(
            query=self.query,
            memory_names=self.mem_name,
            mode="global",
        )
        if len(results) < 360:
            results = await self.tools.get_memory().directly(
                query=self.query,
                memory_names=self.mem_name,
                mode="naive",
            )
        """{
                "answer": consolidated_answer,
                "sources": [
                    {"memory": name, "confidence": score, "content": text},
                    ...
                ]
            }"""
        print(results)
        for r in results:
            print(r)
            self.callback(r)
        return results

    async def extra_query(self, query, **kwargs):
        results = await self.tools.get_memory().directly(
            query=query,
            memory_names=self.mem_name,
            stream=True,
            **kwargs
        )
        print(results)
        for r in results:
            print(r)
            self.callback(r)
        return results

    def generate_mem_name(self):
        return self.tools.get_agent_class("thinkm").mini_task(self.query, "user", "Unique name based on the user query for an memory instance. only the name nothing else!")

    def initialize(self, session_id):
        self.current_session = session_id
        self.mem_name = self.generate_mem_name() + '_' +session_id
        self.init_process_papers()

    async def process(self, query=None) -> Tuple[List[Paper], dict]:
        """Main processing method"""
        if query is not None:
            self.query = query
        t0 = time.process_time()
        queries = self.generate_queries()
        papers = await self.search_and_process_papers(queries)
        print("DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"*12)
        res = await self.generate_insights()
        print(f"Total Presses Time: {time.process_time()-t0:.2f}\nTotal papers Analysed : {len(papers)}/{self.all_ref_papers}")
        return papers, res

async def main(query: str = "Beste strategien in bretspielen sitler von katar"):
    """Main execution function"""
    with Spinner("Init Isaa"):
        tools = get_app("ArXivPDFProcessor", name=None).get_mod("isaa")
        tools.init_isaa(build=True)
    processor = ArXivPDFProcessor(query, tools=tools, limiter=0.5)
    papers, insights = await processor.process()

    print("Generated Insights:", insights)
    print("Generated Insights_list:", processor.last_insights_list)
    await get_app("ArXivPDFProcessor", name=None).a_idle()
    return insights


if __name__ == "__main__":
    asyncio.run(main("Beste strategien AI Agents Development"))



"""

"""
