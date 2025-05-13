import mimetypes
import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    ImageCaptionLoader,
    PythonLoader,
)
from langchain_core.documents import Document

try:

    ImageCaptionLoader_ac = True
except ImportError as e:
    ImageCaptionLoader_ac = e
    ImageCaptionLoader = None
try:
    from langchain_community.document_loaders import BibtexLoader

    BibtexLoader_ac = True
except ImportError as e:
    BibtexLoader_ac = e
    BibtexLoader = None

# try:
from langchain_community.document_loaders import ConcurrentLoader

ConcurrentLoader_ac = True
# except ImportError as e:
#    ConcurrentLoader_ac = e
#    ConcurrentLoader = None
try:
    from langchain_community.document_loaders import UnstructuredImageLoader

    UnstructuredImageLoader_ac = True
except ImportError as e:
    UnstructuredImageLoader_ac = e
    UnstructuredImageLoader = None
try:
    from langchain_community.document_loaders.merge import MergedDataLoader

    MergedDataLoader_ac = True
except ImportError as e:
    MergedDataLoader_ac = e
    MergedDataLoader = None
try:
    #pip install PyMuPDF
    from langchain_community.document_loaders import PyPDFLoader

    PyPDFLoader_ac = True
except ImportError as e:
    PyPDFLoader_ac = e
    PyPDFLoader = None
try:
    from langchain_community.document_loaders import ObsidianLoader

    ObsidianLoader_ac = True
except ImportError as e:
    ObsidianLoader_ac = e
    ObsidianLoader = None
try:
    from langchain_community.document_loaders import UnstructuredODTLoader

    UnstructuredODTLoader_ac = True
except ImportError as e:
    UnstructuredODTLoader_ac = e
    UnstructuredODTLoader = None


def read_bibtex(filename):
    if BibtexLoader_ac is not True:
        raise ImportError(BibtexLoader_ac)
    loader = BibtexLoader(filename)
    docs = loader.load()
    return loader, docs


def python_code_folder_loder(path, glob: str = "**/[!.]*", **kwargs):
    loader = DirectoryLoader(path, glob=glob, loader_cls=PythonLoader, show_progress=True, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def load_from_file_system(path, glob: str = "**/[!.]*", **kwargs):
    if ConcurrentLoader_ac is not True:
        raise ImportError(ConcurrentLoader_ac)
    loader = ConcurrentLoader.from_filesystem(path, glob=glob, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def obsidian_loder(path):
    if ObsidianLoader_ac is not True:
        raise ImportError(ObsidianLoader_ac)
    loader = ObsidianLoader(path)
    def docs():
        return loader.load()
    return loader, docs


def text_loder(text, metadata):
    return Document(page_content=text, metadata=metadata)


def image_loder(image_directory: str, **kwargs):
    if UnstructuredImageLoader_ac is not True:
        raise ImportError(UnstructuredImageLoader_ac)
    loader = UnstructuredImageLoader(image_directory, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def caption_image_urls(list_image_urls: list, **kwargs):
    if ImageCaptionLoader_ac is not True:
        raise ImportError(ImageCaptionLoader_ac)
    loader = ImageCaptionLoader(path_images=list_image_urls, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def pdf_loder(file_path: str, extract_images: bool = False):
    if PyPDFLoader_ac is not True:
        raise ImportError(PyPDFLoader_ac)
    loader = PyPDFLoader(file_path=file_path, extract_images=extract_images)
    def docs():
        return loader.load()
    return loader, docs


def odt_loder(file_path: str, extract_images: bool = False, **kwargs):
    if UnstructuredODTLoader_ac is not True:
        raise ImportError(UnstructuredODTLoader_ac)
    loader = UnstructuredODTLoader(file_path=file_path, extract_images=extract_images, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def merged_data_loader(loaders: list):
    if MergedDataLoader_ac is not True:
        raise ImportError(MergedDataLoader_ac)
    return MergedDataLoader(loaders=loaders)


def route_local_file_to_function(path, **kwargs):
    # Bestimme den Dateityp basierend auf der Dateiendung
    _, ext = os.path.splitext(path)
    mime_type, _ = mimetypes.guess_type(path)

    # Bibtex-Dateien
    if ext.lower() == '.bib':
        return read_bibtex(path)

    # PDF-Dateien
    elif ext.lower() == '.pdf':
        return pdf_loder(path, kwargs.get('extract_images', True))

    # ODT-Dateien
    elif ext.lower() == '.odt':
        return odt_loder(path, kwargs.get('extract_images', True), **kwargs)

    # Bilder
    elif mime_type and mime_type.startswith('image'):
        # Da `image_loder` ein Verzeichnis erwartet, muss hier eine Anpassung erfolgen, falls einzelne Bilder geladen
        # werden sollen.
        return image_loder(os.path.dirname(path), **kwargs)

    # Obsidian Vault
    elif os.path.isdir(path) and any(file.endswith('.md') for file in os.listdir(path)):
        return obsidian_loder(path)

    # Generisches Laden aus dem Dateisystem (für Textdateien oder unbekannte Dateitypen)
    else:
        return load_from_file_system(path, **kwargs)
