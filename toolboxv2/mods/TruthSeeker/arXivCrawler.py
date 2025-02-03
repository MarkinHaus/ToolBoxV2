import threading
import time
from typing import List, Tuple, Optional, Dict
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

from .one import InputProcessor


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
                 download_dir="pdfs"):
        # Initialize components
        self.last_insights_list = None
        self.max_workers = max_workers
        self.tools = tools
        self.semantic_similarity =  InputProcessor().pcs
            #with Spinner("Building agent"):
            #    self.tools.init_isaa(build=True)

        # chunking parameters
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.nsrpq = num_search_result_per_query
        self.max_search = max_search

        # query
        self.query = query
        self.limiter = limiter

        self.temp = True
        self.parser = RobustPDFDownloader(download_dir=download_dir)

        # Ref Papers
        self.all_ref_papers = 0

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

        query_generator = [self.query] + query_generator.queries
        quereis = [sorted_query[0] for sorted_query in sorted([(str_query, self.semantic_similarity(str_query, self.query)) for str_query in query_generator], key=lambda query_tup: query_tup[1], reverse=True)]
        return quereis[:self.max_search]

    def _chunk_document(self, full_text: str) -> List[DocumentChunk]:
        """Split document into manageable chunks"""
        chunks = []
        words = full_text.split()

        try:
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_content = " ".join(chunk_words)

                # Create chunk
                chunk = DocumentChunk(
                    content=chunk_content,
                    page_number=i // self.chunk_size + 1,
                )
                chunks.append(chunk)
        except ValueError:
            chunks = [words]

        return chunks

    def __assess_chunk_relevance(self, chunk: DocumentChunk) -> RelevanceAssessment:
        """Assess relevance of a document chunk"""

        relevance_prompt = f"""
        Evaluate this document chunk's relevance to the query: {self.query}

        Document Chunk:
        {chunk.content[:1000]}...

        Provide:
        1. A relevance score (0-1)
        2. Key sections that are most relevant to the query

        IMPORTANT: Be precise and concise.
        """

        return self.tools.format_class(RelevanceAssessment, relevance_prompt)

    def _assess_chunk_relevance(self, chunk: DocumentChunk, query=None) -> float:
        """Assess relevance of a document chunk"""
        query = query if query else self.query
        return self.semantic_similarity(query, chunk.content)

    def search_and_process_papers(self, queries: List[str]) -> List[Paper]:
        """Search ArXiv, download PDFs, and process them in parallel batches."""
        all_papers = []

        from toolboxv2.mods.isaa.subtools.web_loder import get_pdf_from_url
        lock = threading.Lock()

        def process_query(query):
            papers_for_query = []
            # Search papers
            papers = search_papers(query, max_results=self.nsrpq)
            with lock:
                self.all_ref_papers += len(papers)

            for paper in papers:
                try:
                    # Download PDF
                    pdf_path = self.parser.download_pdf(paper.pdf_url, 'temp_pdf'+paper.title[:20])
                    pdf_pages = self.parser.extract_text_from_pdf(pdf_path)
                    if self.temp:
                        os.remove(pdf_path)
                except Exception as e:
                    print(f"Error processing PDF {paper.pdf_url}: {e}")
                    continue


                # Assess relevance of each chunk
                relevant_chunks = []
                total_relevance = 0
                ref_pages = []
                for page in pdf_pages:
                    # Chunk the document
                    chunks = self._chunk_document(page['text'])
                    for chunk in chunks:
                        relevance_assessment =  sum([self._assess_chunk_relevance(chunk, q) for q in queries])
                        # print(relevance_assessment, chunks.index(chunk))
                        if relevance_assessment >= self.limiter:
                            chunk.relevance_score = relevance_assessment
                            relevant_chunks.append(chunk)
                            total_relevance += relevance_assessment
                            ref_pages.append(page['page_number'])

                # Only add paper if it has relevant chunks
                if relevant_chunks:
                    paper_obj = Paper(
                        title=paper.title,
                        summary=paper.summary,
                        pdf_url=paper.pdf_url,
                        chunks=relevant_chunks,
                        ref_pages=ref_pages,
                        overall_relevance_score=total_relevance / len(relevant_chunks)
                    )
                    papers_for_query.append(paper_obj)

            return papers_for_query

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(process_query, query): query for query in queries}

            for future in tqdm(as_completed(future_to_query), total=len(queries), desc="Processing queries"):
                query = future_to_query[future]
                try:
                    result = future.result()
                    all_papers.extend(result)
                except Exception as e:
                    print(f"Error processing query '{query}': {e}")

        return sorted(all_papers, key=lambda x: x.overall_relevance_score, reverse=True)

    def generate_insights(self, papers: List[Paper]) -> Insights:
        """Generate insights using format_class"""
        if papers is None or len(papers) == 0:
            return Insights(
            is_true=None,
            summary = f"No information's Found on the topik '{self.query}'",
            key_point = ""
        )
        batches = [
            f"Title: {p.title}\n"
            f"Summary: {p.summary}\n"
            f"Relevance Score: {p.overall_relevance_score}\n"
            f"Relevant Chunks: {' | '.join([chunk.content[:600] for chunk in p.chunks[:3]])}"
                for p in papers
        ]
        infos = []
        for i in range(0, min(len(batches), 10), max(1, len(batches)//10)):
            infos.append("\n\n".join(batches[i:i+(len(batches)//10)]))
        # Prepare paper information

        if 'InsightsAgent' not in self.tools.config['agents-name-list']:
            insights_agent = self.tools.get_default_agent_builder("think")
            insights_agent.set_amd_name("InsightsAgent")
            plan_agent = self.tools.register_agent(insights_agent)
            self.tools.format_class_generator(Insights, plan_agent)

        self.tools.print(f"Analysing {len(infos)} key infos chunks")

        def helper_info(h_info):
            try:
                return self.tools.format_class(Insights, (
                f"query: {self.query}"

                "Analyze the following research papers and provide comprehensive insights specifically cut for the query."
                "Papers:"
                   f" {h_info}"
                " Instructions:"
                "1. Create a concise summary relevant for the query!"
                "2. Extract key points directly addressing the query"
                "3. Be precise and evidence-based"
                "4. Only provide date from the papers!"
                "5. if the is_true var is not clear set it to null"
                ))
            except Exception as e:
                def is_true_(info_):
                    try:
                        return self.tools.format_class(ISTRUE, info_).value
                    except Exception as e_:
                        return None
                def summary_(info_):
                    try:
                        return self.tools.mini_task_completion(mini_task="Comprehensive summary addressing the query: "+self.query, user_task=info_)
                    except Exception as e_:
                        return " Error " + info_[:500]
                def key_point_(info_):
                    try:
                        return self.tools.mini_task_completion(mini_task="Extract the single Most important findings for this  query: "+self.query, user_task=info_)
                    except Exception as e_:
                        return " Error " + info_[:500]
                return Insights(
                    is_true=is_true_(h_info),
                    summary=summary_(h_info),
                    key_point=key_point_(h_info)
                )

        insights_list: List[Insights] = [
            helper_info(info)
            for info in infos
        ]

        is_true = [b.is_true for b in insights_list if isinstance(b.is_true, bool)]
        if len(is_true) == 0:
            is_true = None
        else:
            is_true = all(is_true)

        self.last_insights_list = insights_list
        print(insights_list)
        return Insights(
            is_true=is_true,
            summary = self.tools.run_agent("self", text="Summaries (Short, Compact, specific, Include Facts and numbers, and relation ships!)  the following text in relevance for this query: "+self.query+ '\n'+'\n\n'.join([i.summary for i in insights_list])),
            key_point =">\n\n<".join(kp.key_point for kp in insights_list)
        )

    def process(self) -> Tuple[List[Paper], Insights]:
        """Main processing method"""
        t0 = time.process_time()
        with Spinner("Generating Queries"):
            queries = self.generate_queries()
        # print(f"Queries : {queries}")
        with Spinner("Processing Papers"):
            papers = self.search_and_process_papers(queries)

        with Spinner("Generating Insights"):
            insights = self.generate_insights(papers)
        print(f"Total Presses Time: {time.process_time()-t0:.2f}\nTotal papers Analysed : {len(papers)}/{self.all_ref_papers}")
        return papers, insights

def main(query: str = "Beste strategien in bretspielen sitler von katar"):
    """Main execution function"""
    with Spinner("Init Isaa"):
        tools = get_app("ArXivPDFProcessor").get_mod("isaa")
        tools.init_isaa(build=True)
    processor = ArXivPDFProcessor(query, tools=tools, limiter=0.5)
    papers, insights = processor.process()

    print("Generated Insights:", insights)
    print("Generated Insights_list:", processor.last_insights_list)
    return insights


if __name__ == "__main__":
    while True:
        main(input("Query:"))

