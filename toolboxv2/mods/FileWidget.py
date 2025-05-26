# FileWidget.py
import mimetypes
import os
import pickle
import re
import json
import secrets  # For generating secure share IDs
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, AsyncGenerator

# Assuming toolboxv2 and its components are in the Python path
from toolboxv2 import App, Result, RequestData, get_app, MainTool
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage

# from toolboxv2.utils.system.session import RequestSession # Replaced by RequestData

# --- Constants ---
MOD_NAME = Name = "FileWidget"
VERSION = "0.1.0"  # Incremented version
SHARES_METADATA_FILENAME = "filewidget_shares.json"

# --- Module Export ---
# Adjust the module path as per your project structure
# e.g., if FileWidget is in a 'widgets' sub-package: "widgets.FileWidget"
export = get_app(f"widgets.{MOD_NAME}").tb


@dataclass
class ChunkInfo:
    filename: str
    chunk_index: int | None
    total_chunks: int | None
    content: bytes


class MultipartParser:
    def __init__(self, body: bytes):
        self.body = body
        self.boundary = self._extract_boundary()

    def _extract_boundary(self) -> bytes:
        # Erste Zeile enthält die Boundary, muss bytes bleiben
        first_line = self.body.split(b'\r\n', 1)[0]
        return first_line

    def _parse_content_disposition(self, headers_bytes: bytes) -> dict:
        result = {}
        headers = headers_bytes.decode('utf-8', errors='ignore')
        for header_line in headers.split('\r\n'):
            if header_line.lower().startswith('content-disposition'):
                matches = re.findall(r'(\w+)="([^"]+)"', header_line)
                for key, value in matches:
                    result[key] = value
        return result

    def parse(self) -> ChunkInfo:
        if not self.boundary:
            # Fallback or error if boundary is not found (e.g., not multipart)
            # This might happen if the request is not correctly formatted as multipart/form-data
            # For now, let's assume it's a single file if no boundary
            # A more robust parser would handle this better or raise an error
            # For simplicity, we'll try to make it work for single, non-chunked files
            # sent not as multipart but directly. This is a simplification.
            # A proper client would always send multipart for the 'upload' endpoint.
            # Let's assume the request is a direct file upload without form fields for this hack:
            # This part is speculative and depends on how non-chunked, simple uploads are sent.
            # If they are always multipart, this path is not taken.
            # This should ideally raise an error if boundary is expected but not found.
            # For now, let's try to extract filename from request if possible (not standard)
            # and assume the whole body is content. THIS IS A HACK.
            # A real solution would be to enforce multipart or have a separate endpoint.
            # For this exercise, we'll focus on the multipart path.
            # If boundary is not found, it's likely not a valid multipart request for this parser.
            raise ValueError("Invalid multipart request: Boundary not found")

        parts = self.body.split(self.boundary)

        file_content = None
        filename = "unknown_file"  # Default filename
        chunk_index_str = None
        total_chunks_str = None

        # part[0] is usually empty or preamble, part[-1] is the epilogue with --
        # Iterate over actual content parts
        for part_bytes in parts[1:-1]:  # Skip preamble and epilogue
            if not part_bytes.strip():
                continue

            try:
                # Headers are separated from content by \r\n\r\n
                headers_bytes, content_with_crlf = part_bytes.split(b'\r\n\r\n', 1)

                # Content might have a trailing \r\n before the next boundary
                content = content_with_crlf.rsplit(b'\r\n', 1)[0]

                disposition = self._parse_content_disposition(headers_bytes)

                field_name = disposition.get('name')

                if field_name == 'file':
                    file_content = content
                    if disposition.get('filename'):  # Get filename from 'file' part if available
                        filename = disposition.get('filename')
                elif field_name == 'fileName':
                    filename = content.decode('utf-8', errors='ignore')
                elif field_name == 'chunkIndex':
                    chunk_index_str = content.decode('utf-8', errors='ignore')
                elif field_name == 'totalChunks':
                    total_chunks_str = content.decode('utf-8', errors='ignore')
            except ValueError:  # Handles split errors if part is not as expected
                # self.app.logger.warning(f"Could not parse multipart part: {part_bytes[:100]}") # Assuming self.app.logger
                print(f"Warning: Could not parse multipart part: {part_bytes[:100]}")
                continue

        if file_content is None:
            raise ValueError("File content not found in multipart request.")

        return ChunkInfo(
            filename=filename,
            chunk_index=int(chunk_index_str) if chunk_index_str and chunk_index_str.isdigit() else 0,
            # default to 0 for single chunk
            total_chunks=int(total_chunks_str) if total_chunks_str and total_chunks_str.isdigit() else 1,
            # default to 1
            content=file_content
        )


class FileUploadHandler:
    def __init__(self, upload_dir: str = 'uploads'):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, chunk_info: ChunkInfo, storage: BlobStorage) -> str:
        """Speichert die Datei oder Chunk. Chunks werden lokal gespeichert, dann zu BlobStorage gemerged."""
        final_blob_path = chunk_info.filename  # Relative path within BlobStorage

        if chunk_info.total_chunks is None or chunk_info.total_chunks == 1:
            # Komplette Datei direkt in BlobStorage speichern
            with BlobFile(final_blob_path, 'w', storage=storage) as bf:
                bf.write(chunk_info.content)
        else:
            # Chunk lokal speichern
            # Sanitize filename for local path
            safe_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in chunk_info.filename)
            chunk_path = self.upload_dir / f"{safe_filename}.part{chunk_info.chunk_index}"

            with open(chunk_path, 'wb') as f:
                f.write(chunk_info.content)

            if self._all_chunks_received(safe_filename, chunk_info):
                self._merge_chunks_to_blob(safe_filename, chunk_info, final_blob_path, storage)
                self._cleanup_chunks(safe_filename, chunk_info)

        return final_blob_path  # Path within BlobStorage

    def _all_chunks_received(self, safe_filename: str, chunk_info: ChunkInfo) -> bool:
        if chunk_info.total_chunks is None:
            return False
        for i in range(chunk_info.total_chunks):
            chunk_path = self.upload_dir / f"{safe_filename}.part{i}"
            if not chunk_path.exists():
                return False
        return True

    def _merge_chunks_to_blob(self, safe_filename: str, chunk_info: ChunkInfo, final_blob_path: str,
                              storage: BlobStorage):
        with BlobFile(final_blob_path, 'w', storage=storage) as outfile:
            for i in range(chunk_info.total_chunks):
                chunk_path = self.upload_dir / f"{safe_filename}.part{i}"
                with open(chunk_path, 'rb') as chunk_file:
                    outfile.write(chunk_file.read())

    def _cleanup_chunks(self, safe_filename: str, chunk_info: ChunkInfo):
        if chunk_info.total_chunks is None:
            return
        for i in range(chunk_info.total_chunks):
            chunk_path = self.upload_dir / f"{safe_filename}.part{i}"
            if chunk_path.exists():
                os.remove(chunk_path)

class Tools(MainTool):
    def __init__(self, app: App):
        self.upload_handler = None
        self.temp_upload_dir = None
        self.shares = None
        self.shares_metadata_path = None
        self.name = MOD_NAME
        self.version = "0.0.1"
        self.color = "WITHE"
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": MOD_NAME,
            "Version": self.show_version,
        }
        MainTool.__init__(self,
                          load=lambda :None,
                          v=self.version,
                          tool=self.tools,
                          name=self.name,
                          color=self.color,
                          on_exit=self.on_exit)
        self.on_start()
    def on_start(self):
        self.shares_metadata_path = Path(self.app.data_dir) / SHARES_METADATA_FILENAME
        self.shares: Dict[str, Dict[str, Any]] = self._load_shares()

        # Temporary local upload dir for chunks
        self.temp_upload_dir = Path(self.app.data_dir) / "filewidget_tmp_uploads"
        self.temp_upload_dir.mkdir(parents=True, exist_ok=True)
        self.upload_handler = FileUploadHandler(upload_dir=str(self.temp_upload_dir))

        self.app.logger.info(f"{self.name} v{self.version} initialized.")
        self.app.logger.info(f"Shares loaded from: {self.shares_metadata_path}")
        self.app.run_any(("CloudM", "add_ui"),
                    name="FileWidget",
                    title="FileWidget",
                    path=f"/api/FileWidget/ui",
                    description="file management", auth=True
                    )
        self.app.logger.info("Starting FileWidget")

    def on_exit(self):
        self.app.logger.info("Closing FileWidget")

    def show_version(self):
        self.app.logger.info("Version: %s", self.version)
        return self.version

    def _load_shares(self) -> Dict[str, Dict[str, Any]]:
        if self.shares_metadata_path.exists():
            try:
                with open(self.shares_metadata_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.app.logger.error(
                    f"Error decoding JSON from {self.shares_metadata_path}. Starting with empty shares.")
                return {}
        return {}

    def _save_shares(self):
        try:
            with open(self.shares_metadata_path, 'w') as f:
                json.dump(self.shares, f, indent=4)
        except IOError:
            self.app.logger.error(f"Error writing shares to {self.shares_metadata_path}.")

    def _generate_share_id(self) -> str:
        return secrets.token_urlsafe(16)

    async def _get_user_uid_from_request(self, request: RequestData) -> Optional[str]:
        # This depends on how user authentication is handled and session data is populated
        # in RequestData by your ToolBoxV2 setup.
        # Assuming request.session exists and has a 'uid' or 'user_id' attribute.
        if hasattr(request, 'session') and request.session and hasattr(request.session, 'uid'):
            return request.session.uid
        if hasattr(request, 'user') and request.user and hasattr(request.user, 'uid'):  # Alternative common pattern
            return request.user.uid
        self.app.logger.warning("Could not determine user UID from request session/user.")
        return None

    async def get_blob_storage(self, request: Optional[RequestData] = None,
                               owner_uid_override: Optional[str] = None) -> BlobStorage:
        user_uid = owner_uid_override
        is_authenticated_user_storage = False

        if not user_uid and request:  # Try to get UID from request if not overridden
            user_uid = await self._get_user_uid_from_request(request)
            is_authenticated_user_storage = True

        if not user_uid:  # No UID from override or request (e.g. anonymous access attempt to non-shared resource)
            # For public files not tied to a specific user (e.g. system assets),
            # you might return a generic public BlobStorage.
            # For user files, this path should ideally not be hit unless it's for a shared file
            # where owner_uid_override *is* provided.
            # If we are here trying to get a user's storage without a UID, it's an issue.
            if is_authenticated_user_storage:  # This implies an issue with getting UID for an authenticated action
                self.app.logger.error(
                    "Attempted to get user blob storage for an authenticated action, but no UID found.")
                raise ValueError("User not authenticated or UID not found in session for storage access.")
            else:  # Generic public storage if no specific user context
                self.app.logger.info("Accessing generic public BlobStorage.")
                return BlobStorage(Path(self.app.data_dir) / 'public_files', 0)

        # User-specific storage
        # if user_uid not in self.blob_storage_cache: # Not caching instances to avoid state issues with BlobStorage itself
        storage_path = Path(self.app.data_dir) / 'user_storages' / user_uid
        # self.blob_storage_cache[user_uid] = BlobStorage(storage_path)
        # self.app.logger.debug(f"BlobStorage instance created/retrieved for user {user_uid} at {storage_path}")
        return BlobStorage(storage_path)
        # return self.blob_storage_cache[user_uid]


def get_template_content() -> str:
    # Content from the original get_template method
    # Added a share button placeholder and an input for the share link
    return """
    <title>File Manager</title>
    <style>
    .tree-view { color: var(--theme-bg);font-family: monospace; margin: 10px 0; border: 1px solid #ddd; padding: 10px; max-height: 600px; overflow-y: auto; background: #f8f9fa; }
    .tree-view p { color: var(--theme-bg)}
    .folder-group { font-weight: bold; color: #2c3e50; padding: 5px; margin-top: 10px; background: #edf2f7; border-radius: 4px; cursor: pointer; }
    .group-content { margin-left: 20px; border-left: 2px solid #e2e8f0; padding-left: 10px; display: none; }
    .folder { cursor: pointer; padding: 2px 5px; margin: 2px 0; color: #4a5568; }
    .folder:hover { background: #edf2f7; border-radius: 4px; }
    .file { padding: 2px 5px; margin: 2px 0; color: #718096; display: flex; justify-content: space-between; align-items: center; }
    .file:hover { background: #edf2f7; border-radius: 4px; color: #2d3748; }
    .file-name { cursor: pointer; flex-grow: 1; }
    .share-btn { margin-left: 10px; padding: 2px 5px; background-color: #3498db; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 0.8em; }
    .share-btn:hover { background-color: #2980b9; }
    .folder-content { margin-left: 20px; border-left: 1px solid #e2e8f0; padding-left: 10px; display: none; }
    .folder-content.open, .group-content.open { display: block; }
    .folder::before, .folder-group::before { content: '▶'; display: inline-block; margin-right: 5px; transition: transform 0.2s; }
    .folder.open::before, .folder-group.open::before { transform: rotate(90deg); }
    .drop-zone { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 10px 0; cursor: pointer; }
    .drop-zone.dragover { background-color: #e1e1e1; border-color: #999; }
    .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden; margin-top: 5px; }
    .progress { width: 0%; height: 100%; background-color: #4CAF50; transition: width 0.3s ease-in-out; }
    #shareLinkContainer { margin-top: 15px; padding: 10px; background-color: #e9ecef; border-radius: 4px; display: none; }
    #shareLinkInput { width: calc(100% - 80px); padding: 8px; border: 1px solid #ced4da; border-radius: 4px; margin-right: 5px; }
    #copyShareLinkBtn { padding: 8px 12px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>

    <div class="file-container">
        <h2>File Manager</h2>
        <div class="drop-zone" id="dropZone">
            <p>Drag & Drop files here or click to upload</p>
            <input type="file" id="fileInput" multiple style="display: none;">
        </div>
        <div class="progress-bar" style="display: none;"><div class="progress" id="uploadProgress"></div></div>

        <div id="shareLinkContainer">
            <input type="text" id="shareLinkInput" readonly>
            <button id="copyShareLinkBtn">Copy</button>
        </div>

        <div class="tree-view" id="fileTree">Loading file tree...</div>
    </div>

    <script>
        // Ensure TB (ToolBox Client-Side) is available, or use standard fetch
        // For this example, we'll use standard fetch for API calls.
        // const TB_API_PREFIX = '/api/FileWidget'; // Adjust if your API routes are different

        class FileManager {
            constructor() {
                this.dropZone = document.getElementById('dropZone');
                this.fileInput = document.getElementById('fileInput');
                this.fileTree = document.getElementById('fileTree');
                this.progressBar = document.querySelector('.progress-bar');
                this.progress = document.getElementById('uploadProgress');
                this.shareLinkContainer = document.getElementById('shareLinkContainer');
                this.shareLinkInput = document.getElementById('shareLinkInput');
                this.copyShareLinkBtn = document.getElementById('copyShareLinkBtn');

                this.responseCache = {}; // To store original file paths from server

                this.initEventListeners();
                this.initLoadFileTree();
            }

            initEventListeners() {
                this.dropZone.addEventListener('click', () => this.fileInput.click());
                this.fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));
                this.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); this.dropZone.classList.add('dragover'); });
                this.dropZone.addEventListener('dragleave', () => this.dropZone.classList.remove('dragover'));
                this.dropZone.addEventListener('drop', (e) => { e.preventDefault(); this.dropZone.classList.remove('dragover'); this.handleFiles(e.dataTransfer.files); });
                this.copyShareLinkBtn.addEventListener('click', () => this.copyShareLink());
            }

            async handleFiles(files) {
                for (const file of files) {
                    await this.uploadFile(file);
                }
                this.loadFileTree(); // Refresh tree after upload
            }

            async uploadFile(file) {
                this.progressBar.style.display = 'block';
                this.progress.style.width = '0%';

                const chunkSize = 1024 * 1024; // 1MB
                const totalChunks = Math.ceil(file.size / chunkSize);

                for (let i = 0; i < totalChunks; i++) {
                    const chunk = file.slice(i * chunkSize, (i + 1) * chunkSize);
                    const formData = new FormData();
                    formData.append('file', chunk, file.name); // Ensure filename is passed with the blob
                    formData.append('fileName', file.name);
                    formData.append('chunkIndex', i);
                    formData.append('totalChunks', totalChunks);

                    try {
                        const response = await fetch('/api/FileWidget/upload', { method: 'POST', body: formData });
                        if (!response.ok) {
                            const errorData = await response.json();
                            console.error('Upload chunk failed:', errorData.info?.help_text || 'Unknown error');
                            alert('Upload failed: ' + (errorData.info?.help_text || 'Server error'));
                            this.progressBar.style.display = 'none';
                            return;
                        }
                    } catch (error) {
                        console.error('Network error during upload:', error);
                        alert('Upload failed: Network error');
                        this.progressBar.style.display = 'none';
                        return;
                    }
                    const progressVal = ((i + 1) / totalChunks) * 100;
                    this.progress.style.width = progressVal + '%';
                }
                setTimeout(() => { this.progressBar.style.display = 'none'; this.progress.style.width = '0%'; }, 1000);
            }

            initLoadFileTree() {
                // Delay slightly to ensure DOM is ready or TB is initialized if it were used
                setTimeout(() => this.loadFileTree(), 500);
            }

            async loadFileTree() {
                try {
                    this.fileTree.innerHTML = '<i>Loading...</i>';
                    const response = await fetch('/api/FileWidget/files');
                    if (!response.ok) {
                        this.fileTree.innerHTML = '<p style="color:red;">Error loading files.</p>';
                        console.error("Error loading file tree, status:", response.status);
                        return;
                    }
                    const apiResponse = await response.json();
                     // Assuming the actual tree data is in apiResponse.result.data
                    if (apiResponse && apiResponse.result && apiResponse.result.data) {
                        this.renderFileTree(apiResponse.result.data);
                    } else {
                        this.fileTree.innerHTML = '<p>No files found or invalid response structure.</p>';
                        console.warn("File tree data not in expected format:", apiResponse);
                    }
                } catch (error) {
                    this.fileTree.innerHTML = '<p style="color:red;">Failed to fetch file tree.</p>';
                    console.error("Error in loadFileTree:", error);
                }
            }

            renderFileTree(treeData) {
                this.responseCache = {}; // Clear cache before rendering
                this.fileTree.innerHTML = this.buildTreeHTML(treeData);
                if (!this.fileTree.innerHTML) {
                    this.fileTree.innerHTML = "<p>No files or folders.</p>";
                }
                this.addTreeEventListeners();
            }

            buildTreeHTML(node, currentPath = '') {
                let html = '';
                const entries = Object.entries(node).sort(([keyA, valA], [keyB, valB]) => {
                    const isDirA = typeof valA === 'object';
                    const isDirB = typeof valB === 'object';
                    if (isDirA !== isDirB) return isDirA ? -1 : 1; // Directories first
                    return keyA.localeCompare(keyB); // Then alphanumeric
                });

                for (const [name, content] of entries) {
                    const fullPathKey = currentPath ? `${currentPath}/${name}` : name;
                    if (typeof content === 'object' && content !== null) { // It's a folder
                        html += `<div class="folder" data-folder-path="${fullPathKey}">📁 ${name}</div>`;
                        html += `<div class="folder-content">`;
                        html += this.buildTreeHTML(content, fullPathKey);
                        html += `</div>`;
                    } else { // It's a file, content is the actual path for download/share
                        const originalFilePath = content; // This is the important part from server
                        this.responseCache[fullPathKey] = originalFilePath; // Store for later use
                        const icon = this.getFileIcon(name);
                        html += `<div class="file" data-display-path="${fullPathKey}">
                                    <span class="file-name" data-file-path="${fullPathKey}">${icon} ${name}</span>
                                    <button class="share-btn" data-share-path="${fullPathKey}">Share</button>
                                 </div>`;
                    }
                }
                return html;
            }

            getFileIcon(filename) {
                const ext = filename.split('.').pop()?.toLowerCase() || 'default';
                const iconMap = {'agent':'🤖','json':'📋','pkl':'📦','txt':'📝','data':'💾','ipy':'🐍','bin':'📀','sqlite3':'🗄️','vec':'📊','pickle':'🥒','html':'🌐','js':'📜','md':'📑','py':'🐍','default':'📄', 'png':'🖼️', 'jpg':'🖼️', 'jpeg':'🖼️', 'gif':'🖼️', 'pdf':'📕', 'zip':'📦'};
                return iconMap[ext] || iconMap['default'];
            }

            addTreeEventListeners() {
                this.fileTree.querySelectorAll('.file-name').forEach(fileEl => {
                    fileEl.addEventListener('click', (e) => {
                        const displayPath = e.currentTarget.dataset.filePath;
                        const actualPath = this.responseCache[displayPath];
                        if (actualPath) {
                            this.downloadFile(actualPath);
                        } else {
                            console.error("Actual path not found for displayed path:", displayPath);
                            alert("Could not determine file path for download.");
                        }
                    });
                });
                this.fileTree.querySelectorAll('.share-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const displayPath = e.currentTarget.dataset.sharePath;
                         const actualPath = this.responseCache[displayPath];
                        if (actualPath) {
                            this.createShareLink(actualPath);
                        } else {
                            console.error("Actual path not found for displayed path:", displayPath);
                            alert("Could not determine file path for sharing.");
                        }
                    });
                });
                this.fileTree.querySelectorAll('.folder, .folder-group').forEach(folder => {
                    folder.addEventListener('click', (e) => {
                        e.stopPropagation();
                        folder.classList.toggle('open');
                        const content = folder.nextElementSibling;
                        if (content && (content.classList.contains('folder-content') || content.classList.contains('group-content'))) {
                            content.classList.toggle('open');
                        }
                    });
                });
            }

            async downloadFile(actualFilePathFromServer) {
                // actualFilePathFromServer is the path BlobStorage knows, like "myfile.txt" or "folder/myfile.txt"
                const response = await fetch(`/api/FileWidget/download?path=${encodeURIComponent(actualFilePathFromServer)}`);
                if (!response.ok) {
                    alert('Error downloading file.');
                    console.error("Download failed", response.status);
                    return;
                }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = actualFilePathFromServer.split('/').pop(); // Get filename from path
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }

            async createShareLink(actualFilePathFromServer) {
                try {
                    const response = await fetch(`/api/FileWidget/create_share_link?file_path=${encodeURIComponent(actualFilePathFromServer)}&share_type=public`);
                    if (!response.ok) {
                        const errorData = await response.json();
                        alert('Failed to create share link: ' + (errorData.info?.help_text || 'Server error'));
                        return;
                    }
                    const result = await response.json();
                    if (result.result && result.result.data && result.result.data.share_link) {
                        this.shareLinkInput.value = result.result.data.share_link;
                        this.shareLinkContainer.style.display = 'block';
                    } else {
                        alert('Could not retrieve share link from server response.');
                    }
                } catch (error) {
                    console.error("Error creating share link:", error);
                    alert('Error creating share link. Check console.');
                }
            }

            copyShareLink() {
                this.shareLinkInput.select();
                document.execCommand('copy');
                alert('Share link copied to clipboard!');
            }
        }
        // Ensure this script runs after the DOM is fully loaded,
        // or if TB framework has an event for widget initialization, use that.
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => new FileManager());
        } else {
            new FileManager();
        }
    </script>
    """


# --- API Endpoints using @export ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, name="ui", api_methods=['GET'])
async def get_main_ui(self) -> Result:
    """Serves the main HTML UI for the FileWidget."""
    # Here, you'd typically use the app's template rendering mechanism
    # if it's more complex or involves BaseWidget features.
    # For simplicity, returning raw HTML via Result.html.
    # The `unsave="true"` in the script tag from original is `unSave="true"`.
    # Modern browsers/HTMX might handle script execution differently.
    # Standard script tags in HTML served this way should execute.
    html_content = get_template_content()
    return Result.html(data=html_content)

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="upload", api_methods=['POST'], request_as_kwarg=True)
async def handle_upload(self, request: RequestData) -> Result:
    if not request.body:
        return Result.default_user_error(info="No data received for upload.", exec_code=400)

    try:
        storage = await self.get_blob_storage(request)
        parser = MultipartParser(request.body)
        chunk_info = parser.parse()

        # Ensure filename is sensible
        if not chunk_info.filename or chunk_info.filename == "unknown_file":
            return Result.default_user_error(info="Filename not provided or invalid in upload.", exec_code=400)

        # Sanitize filename to prevent path traversal issues, BlobFile should handle this too
        # but good to be cautious. For BlobStorage, the key itself is usually sanitized or managed.
        # The path here is relative to the user's storage root.
        blob_relative_path = Path(chunk_info.filename).name  # Use only the filename part for security

        # Update chunk_info with the sanitized/final path name
        chunk_info_to_save = ChunkInfo(
            filename=blob_relative_path,  # This is the key in BlobStorage
            chunk_index=chunk_info.chunk_index,
            total_chunks=chunk_info.total_chunks,
            content=chunk_info.content
        )

        saved_blob_path = self.upload_handler.save_file(chunk_info_to_save, storage)
        self.app.logger.info(
            f"File '{saved_blob_path}' uploaded/chunk saved by user (UID from session if available).")
        return Result.ok(data={"message": "File uploaded successfully", "path": saved_blob_path})
    except ValueError as e:  # Catch specific errors from parser or handler
        self.app.logger.error(f"Upload processing error: {e}", exc_info=True)
        return Result.default_user_error(info=f"Upload error: {str(e)}", exec_code=400)
    except Exception as e:
        self.app.logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
        return Result.default_internal_error(info="An unexpected error occurred during upload.")

async def _prepare_file_response(self, storage: BlobStorage, blob_path: str) -> Result:
    try:
        # Sanitize blob_path, prevent '..' (BlobFile might do this, but good practice)
        # For BlobStorage, paths are usually relative to its root.
        # Path(blob_path).is_absolute() or ".." in blob_path could be checks here.
        # However, BlobFile itself should handle the sandboxing within its storage_root.

        if not storage.exists(blob_path):
            self.app.logger.warning(f"File not found in BlobStorage: {blob_path}")
            return Result.default_user_error(info="File not found.", exec_code=404)

        filename = Path(blob_path).name
        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = 'application/octet-stream'

        # For streaming large files, you'd implement BlobFile.stream() or read in chunks
        # For now, reading whole file (as in original)
        file_size = storage.get_size(blob_path)  # Get size for Content-Length

        # Streamer function for Result.binary
        async def file_streamer() -> AsyncGenerator[bytes, None]:
            with BlobFile(blob_path, 'r', storage=storage) as bf:
                # Read in chunks if you want to support large files better
                # For now, matching original behavior of reading all then sending
                # However, Result.binary with a streamer is better.
                # Let's assume BlobFile can be read in chunks or bf.read() is efficient enough
                # For a true stream:
                # chunk_size = 1024 * 1024 # 1MB
                # while True:
                #     data_chunk = bf.read(chunk_size)
                #     if not data_chunk:
                #         break
                #     yield data_chunk
                # For simplicity, if bf.read() returns all data:
                yield bf.read()

        return Result.binary(
            stream_generator=file_streamer(),
            content_type=content_type,
            download_name=filename,
            content_length=file_size  # Important for clients
        )
    except FileNotFoundError:  # Should be caught by storage.exists earlier
        self.app.logger.warning(f"Download attempt for non-existent file: {blob_path}")
        return Result.default_user_error(info="File not found.", exec_code=404)
    except Exception as e:
        self.app.logger.error(f"Error processing download for {blob_path}: {e}", exc_info=True)
        return Result.default_internal_error(info="Error processing download.")

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="download", api_methods=['GET'], request_as_kwarg=True)
async def handle_download(self, request: RequestData) -> Result:
    blob_path = request.query_params.get('path')
    if not blob_path:
        return Result.default_user_error(info="File path parameter is missing.", exec_code=400)

    try:
        storage = await self.get_blob_storage(request)  # User's own storage
        self.app.logger.info(f"User download request for: {blob_path}")
        return await self._prepare_file_response(storage, blob_path)
    except ValueError as e:  # e.g. user not authenticated for storage access
        self.app.logger.warning(f"Auth error during download for path {blob_path}: {e}")
        return Result.default_user_error(info=str(e), exec_code=401)

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="files", api_methods=['GET'], request_as_kwarg=True)
async def get_file_tree(self, request: RequestData) -> Result:
    try:
        storage = await self.get_blob_storage(request)

        # The original `get_file_tree` logic for building the tree structure
        # by iterating through blob_ids and parsing their metadata is specific
        # to how BlobStorage internally stores file listings or if it relies on pickled metadata.
        # A more typical BlobStorage might have a list_files() method.
        # Assuming the original logic is what's needed for this BlobStorage implementation.

        tree: Dict[str, Any] = {}
        # This part depends heavily on BlobStorage internals.
        # If BlobStorage._get_all_blob_ids() and reading pickled metadata is the way, use it.
        # A simpler BlobStorage might offer storage.list_files(recursive=True)
        try:
            # This is the complex part from the original, depends on BlobStorage's design
            # If BlobStorage stores files with their full paths as IDs (e.g. "folder/file.txt")
            # and has a method like `storage.list_all_paths()`
            all_paths = []
            if hasattr(storage, '_get_all_blob_ids_and_paths'):  # Idealized method
                all_paths = storage._get_all_blob_ids_and_paths()  # Returns list of relative paths
            elif hasattr(storage, '_get_all_blob_ids'):  # Original approach
                blob_ids = storage._get_all_blob_ids()
                for blob_id in blob_ids:  # blob_id is often the filename/path itself
                    # The original code tries to load pickled data for each blob_id to find paths.
                    # This seems overly complex if blob_id is already the path.
                    # Let's assume blob_id IS the relative path for simplicity.
                    # If not, the original's pickle loading logic needs to be here.
                    # For now, let's assume `blob_id` is the path.
                    # If it's an opaque ID, we need a way to get path from ID or metadata.
                    # The simplest interpretation for a file storage is that the ID *is* the path.
                    all_paths.append(str(blob_id))  # Assuming blob_id is path
            else:  # Fallback: try to list based on directory structure if possible (not standard for all blob storages)
                if storage.storage_root and Path(storage.storage_root).exists():
                    for item in Path(storage.storage_root).rglob('*'):
                        if item.is_file():
                            all_paths.append(str(item.relative_to(storage.storage_root)))

            for file_path_str in all_paths:
                path_parts = file_path_str.split('/')
                current_level = tree
                for i, part in enumerate(path_parts):
                    if not part: continue
                    if i == len(path_parts) - 1:  # It's a file
                        current_level[part] = file_path_str  # Store the full path as value
                    else:  # It's a folder
                        if part not in current_level:
                            current_level[part] = {}
                        elif not isinstance(current_level[part], dict):  # Conflict: file with same name as folder
                            self.app.logger.warning(
                                f"File/folder name conflict for '{part}' in path '{file_path_str}'")
                            # Decide on handling: overwrite, skip, or error
                            # For now, skip making it a dict if it's already a file path
                            break
                        current_level = current_level[part]

        except Exception as e:
            self.app.logger.error(f"Error building file tree from BlobStorage: {e}", exc_info=True)
            return Result.default_internal_error(info="Could not list files.")

        self.app.logger.debug(f"File tree for user: {tree}")
        return Result.json(data=tree)
    except ValueError as e:  # e.g. user not authenticated for storage access
        self.app.logger.warning(f"Auth error during file tree access: {e}")
        return Result.default_user_error(info=str(e), exec_code=401)
    except Exception as e:
        self.app.logger.error(f"Unexpected error in get_file_tree: {e}", exc_info=True)
        return Result.default_internal_error(info="Failed to retrieve file list.")

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="create_share_link", api_methods=['GET', 'POST'],
        request_as_kwarg=True)
async def create_share_link(self, request: RequestData) -> Result:
    user_uid = await self._get_user_uid_from_request(request)
    if not user_uid:
        return Result.default_user_error(info="Authentication required to create share links.", exec_code=401)

    file_path = request.query_params.get('file_path') if request.method == 'GET' else (
            await request.json_body() or {}).get('file_path')
    # share_type = request.query_params.get('share_type', 'link') # 'public' or 'link' for now same

    if not file_path:
        return Result.default_user_error(info="Parameter 'file_path' is required.", exec_code=400)

    # Validate that the user owns this file path
    user_storage = await self.get_blob_storage(request)  # Gets storage for the authenticated user
    if not user_storage.exists(file_path):
        return Result.default_user_error(info=f"File not found in your storage: {file_path}", exec_code=404)

    share_id = self._generate_share_id()
    self.shares[share_id] = {
        "owner_uid": user_uid,
        "file_path": file_path,
        "created_at": time.time(),  # Assuming app has a timestamp utility
        # "share_type": share_type
    }
    self._save_shares()

    # Construct the full share link URL
    # This depends on your server's domain and how API routes are exposed.
    # Assuming standard /api/ModuleName/function_name structure.
    base_url = request.base_url  # e.g. "http://localhost:8000"
    # Ensure base_url doesn't have trailing slash, and API path starts with one
    share_access_path = f"/api/{MOD_NAME}/shared/{share_id}"
    full_share_link = str(base_url).rstrip('/') + share_access_path

    self.app.logger.info(f"Share link created by UID {user_uid} for path '{file_path}': {share_id}")
    return Result.ok(data={"share_id": share_id, "share_link": full_share_link})

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="shared", api_methods=['GET'],
        request_as_kwarg=True)
async def access_shared_file(self, request: RequestData, share_id: str) -> Result:
    """
    Accesses a shared file via its share_id.
    The URL for this would be like /api/FileWidget/shared/{share_id_value}
    ` in @export tells ToolBoxV2 to extract 'share_id' from the path.
    """
    if not share_id:  # Should be caught by routing  is mandatory
        return Result.default_user_error(info="Share ID is missing.", exec_code=400)

    share_info = self.shares.get(share_id)
    if not share_info:
        return Result.default_user_error(info="Share link is invalid or has expired.", exec_code=404)

    owner_uid = share_info["owner_uid"]
    file_path = share_info["file_path"]

    try:
        # Get BlobStorage for the owner, not the current request's user (if any)
        owner_storage = await self.get_blob_storage(owner_uid_override=owner_uid)
        self.app.logger.info(f"Accessing shared file via link {share_id}: owner {owner_uid}, path {file_path}")
        return await self._prepare_file_response(owner_storage, file_path)
    except Exception as e:
        self.app.logger.error(f"Error accessing shared file {share_id} (owner {owner_uid}, path {file_path}): {e}",
                              exc_info=True)
        return Result.default_internal_error(info="Could not retrieve shared file.")

# To make this runnable/testable with a ToolBoxV2 app, you'd typically have an app setup:
#
# from toolboxv2 import App
#
# if __name__ == "__main__":
#     # This is a mock app setup for standalone thinking.
#     # In a real ToolBoxV2 app, the App instance is managed globally or passed around.
#     class MockApp(App):
#         def __init__(self):
#             self.data_dir = Path("./.temp_tb_data") # Example data directory
#             self.data_dir.mkdir(exist_ok=True)
#             # Mock logger
#             class MockLogger:
#                 def info(self, msg): print(f"INFO: {msg}")
#                 def error(self, msg, exc_info=None): print(f"ERROR: {msg}")
#                 def warning(self, msg): print(f"WARN: {msg}")
#                 def debug(self, msg): print(f"DEBUG: {msg}")
#             self.logger = MockLogger()
#         def get_timestamp(self): import time; return time.time()

#     mock_app_instance = MockApp()
#     # The FileWidget would be instantiated by the ToolBoxV2 framework,
#     # likely when its module is loaded or a request targets it.
#     # file_widget_instance = FileWidget(app=mock_app_instance)
#
#     # Then, ToolBoxV2's routing would call the exported methods based on requests.
#     # e.g., a GET to /api/FileWidget/ui would call file_widget_instance.get_main_ui(...)
