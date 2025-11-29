# FileWidget.py
import base64  # Added for base64 decoding
import json
import mimetypes
import os
import secrets  # For generating secure share IDs
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Assuming toolboxv2 and its components are in the Python path
from toolboxv2 import App, MainTool, RequestData, Result, get_app
from toolboxv2.utils.extras.base_widget import get_current_user_from_request
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage

# --- Constants ---
MOD_NAME = Name = "FileWidget"
VERSION = "0.2.1"  # Incremented version
SHARES_METADATA_FILENAME = "filewidget_shares.json"

# --- Module Export ---
export = get_app(f"widgets.{MOD_NAME}").tb


@dataclass
class ChunkInfo:
    filename: str  # This is the final intended filename
    chunk_index: int
    total_chunks: int
    content: bytes  # Raw bytes of the chunk


class FileUploadHandler:
    def __init__(self, upload_dir: str = 'uploads'):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # self.app = get_app().app # If logger is needed here

    def save_file(self, chunk_info: ChunkInfo, storage: BlobStorage) -> str:
        """Speichert die Datei oder Chunk. Chunks werden lokal gespeichert, dann zu BlobStorage gemerged."""
        final_blob_path = Path(chunk_info.filename).name  # Use only filename part for security within blob storage

        if chunk_info.total_chunks == 1:
            # Komplette Datei direkt in BlobStorage speichern
            # print(f"Saving single part file: {final_blob_path} to BlobStorage directly.") # Debug
            with BlobFile(final_blob_path, 'w', storage=storage) as bf:
                bf.write(chunk_info.content)
        else:
            # Chunk lokal speichern
            # Sanitize filename for local path (original chunk_info.filename might contain path parts client-side)
            safe_base_filename = "".join(
                c if c.isalnum() or c in ('.', '_', '-') else '_' for c in Path(chunk_info.filename).name)
            chunk_path = self.upload_dir / f"{safe_base_filename}.part{chunk_info.chunk_index}"
            # print(f"Saving chunk: {chunk_path} locally. Total chunks: {chunk_info.total_chunks}") # Debug

            with open(chunk_path, 'wb') as f:
                f.write(chunk_info.content)

            if self._all_chunks_received(safe_base_filename, chunk_info.total_chunks):
                # print(f"All chunks received for {safe_base_filename}. Merging to BlobStorage path: {final_blob_path}") # Debug
                self._merge_chunks_to_blob(safe_base_filename, chunk_info.total_chunks, final_blob_path, storage)
                self._cleanup_chunks(safe_base_filename, chunk_info.total_chunks)
            # else:
            # print(f"Still waiting for more chunks for {safe_base_filename}.") # Debug

        return final_blob_path  # Path within BlobStorage

    def _all_chunks_received(self, safe_base_filename: str, total_chunks: int) -> bool:
        for i in range(total_chunks):
            chunk_path = self.upload_dir / f"{safe_base_filename}.part{i}"
            if not chunk_path.exists():
                # print(f"Chunk {i} for {safe_base_filename} not found. Path: {chunk_path}") # Debug
                return False
        # print(f"All {total_chunks} chunks found for {safe_base_filename}.") # Debug
        return True

    def _merge_chunks_to_blob(self, safe_base_filename: str, total_chunks: int, final_blob_path: str,
                              storage: BlobStorage):
        # print(f"Merging {total_chunks} chunks for {safe_base_filename} into Blob: {final_blob_path}") # Debug
        with BlobFile(final_blob_path, 'w', storage=storage) as outfile:
            for i in range(total_chunks):
                chunk_path = self.upload_dir / f"{safe_base_filename}.part{i}"
                # print(f"Appending chunk {i} ({chunk_path}) to Blob.") # Debug
                with open(chunk_path, 'rb') as chunk_file:
                    outfile.write(chunk_file.read())
        # print(f"Finished merging chunks for {safe_base_filename} to Blob: {final_blob_path}") # Debug

    def _cleanup_chunks(self, safe_base_filename: str, total_chunks: int):
        # print(f"Cleaning up {total_chunks} chunks for {safe_base_filename}.") # Debug
        for i in range(total_chunks):
            chunk_path = self.upload_dir / f"{safe_base_filename}.part{i}"
            if chunk_path.exists():
                # print(f"Removing chunk: {chunk_path}") # Debug
                try:
                    os.remove(chunk_path)
                except OSError as e:
                    # self.app.logger.error(f"Error removing chunk {chunk_path}: {e}") # If logger available
                    print(f"Error removing chunk {chunk_path}: {e}")


class Tools(MainTool):
    def __init__(self, app: App):
        self.upload_handler = None
        self.temp_upload_dir = None
        self.shares = None
        self.shares_metadata_path = None
        self.name = MOD_NAME
        self.version = VERSION  # Use class constant
        self.color = "WHITE"  # Corrected casing
        self.tools = {
            "all": [["Version", "Shows current Version"]],
            "name": MOD_NAME,
            "Version": self.show_version,
        }
        MainTool.__init__(self,
                          load=lambda: None,
                          v=self.version,
                          tool=self.tools,
                          name=self.name,
                          color=self.color,
                          on_exit=self.on_exit)

    def on_start(self):
        self.shares_metadata_path = Path(self.app.data_dir) / SHARES_METADATA_FILENAME
        self.shares: dict[str, dict[str, Any]] = self._load_shares()

        self.temp_upload_dir = Path(self.app.data_dir) / "filewidget_tmp_uploads"
        self.temp_upload_dir.mkdir(parents=True, exist_ok=True)
        self.upload_handler = FileUploadHandler(upload_dir=str(self.temp_upload_dir))

        self.app.logger.info(f"{self.name} v{self.version} initialized.")
        self.app.logger.info(f"Shares loaded from: {self.shares_metadata_path}")
        self.app.logger.info(f"Temporary upload directory: {self.temp_upload_dir}")

        self.app.run_any(("CloudM", "add_ui"),
                         name="FileWidget",
                         title="FileWidget",
                         path="/api/FileWidget/ui",
                         # This path is where CloudM lists it, not necessarily where users access it directly
                         description="file management", auth=True
                         )
        self.app.logger.info("FileWidget UI registered with CloudM.")

    def on_exit(self):
        self.app.logger.info(f"Closing {self.name}")

    def show_version(self):
        self.app.logger.info(f"{self.name} Version: {self.version}")
        return self.version

    def _load_shares(self) -> dict[str, dict[str, Any]]:
        if self.shares_metadata_path is not None and self.shares_metadata_path.exists():
            try:
                with open(self.shares_metadata_path) as f:
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
        except OSError:
            self.app.logger.error(f"Error writing shares to {self.shares_metadata_path}.")

    def _generate_share_id(self) -> str:
        return secrets.token_urlsafe(16)

    async def _get_user_uid_from_request(self, request: RequestData) -> str | None:
        user = await get_current_user_from_request(self.app, request)
        if user and hasattr(user, 'uid') and user.uid:  # Ensure uid attribute exists and is not empty
            return user.uid
        # Fallback: try to get 'SiID' or 'user_name' from session if user object is not fully formed
        # This depends on how get_current_user_from_request populates the User object
        # or if request.session is directly usable.
        if request and request.session:
            # Assuming session is a dict-like object as populated by Rust via kwargs['request']['session']
            # 'SiID' is often used as a unique session/user identifier in such systems.
            uid_from_session = request.session.SiID or (request.session.get('uid') if hasattr(request.session, 'get') else None)
            if uid_from_session:
                self.app.logger.debug(f"Retrieved UID '{uid_from_session}' from request.session as fallback.")
                return uid_from_session
            self.app.logger.warning(
                f"User object found but no UID. Session keys: {list(request.session.keys()) if request.session else 'No session'}")
        return None

    async def get_blob_storage(self, request: RequestData | None = None,
                               owner_uid_override: str | None = None) -> BlobStorage:
        user_uid: str | None = None

        if owner_uid_override:
            user_uid = owner_uid_override
            self.app.logger.debug(f"BlobStorage access for overridden UID: {user_uid}")
        elif request:
            user_uid = await self._get_user_uid_from_request(request)
            if not user_uid:
                self.app.logger.warning("Authenticated action attempted, but no user UID found in request session.")
                raise ValueError("User authentication required, or UID not found in session for storage access.")
            self.app.logger.debug(f"BlobStorage access for authenticated user UID: {user_uid}")
        else:
            self.app.logger.error("BlobStorage access attempted without user context (no request and no UID override).")
            raise ValueError("Cannot determine user context for BlobStorage access.")

        # User-specific storage path
        storage_path = Path(self.app.data_dir) / 'user_storages' / user_uid
        # Ensure storage_path is not trying to escape (e.g. if user_uid had '..')
        # Path resolution should handle this, but being explicit is safer if user_uid is not strictly controlled.
        # For now, assume user_uid is a safe directory name.
        return BlobStorage(self.app.root_blob_storage.servers, storage_directory=str(storage_path))


def get_template_content() -> str:
    return """
    <title>File Manager</title>
    <style>
    .tree-view {font-family: monospace; margin: 10px 0; border: 1px solid #ddd; padding: 10px; max-height: 600px; overflow-y: auto; background: var(--theme-bg); }
    .folder-group { font-weight: bold; color: #2c3e50; padding: 5px; margin-top: 10px; background: var(--theme-bg); border-radius: 4px; cursor: pointer; }
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
    .progress-bar-container { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden; margin-top: 5px; display: none; }
    .progress-bar { width: 0%; height: 100%; background-color: #4CAF50; transition: width 0.3s ease-in-out; text-align: center; color: white; line-height:20px; }
    /* Styles for TB.ui.Modal (if not globally available through tb.css) */
    /* Basic modal styles, assuming TB provides more complete ones */
    .tb-modal-backdrop { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1050;}
    .tb-modal-dialog { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 300px; max-width: 500px;}
    .tb-modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px;}
    .tb-modal-title { font-size: 1.25rem; margin: 0;}
    .tb-modal-body { margin-bottom: 15px; }
    .tb-modal-footer { text-align: right; border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px;}
    .tb-modal-footer button { margin-left: 5px; }
    #shareLinkInputModal { width: calc(100% - 22px); padding: 8px; border: 1px solid #ced4da; border-radius: 4px; margin-bottom: 10px; }

    </style>

    <div class="file-container">
        <h2>File Manager</h2>
        <div class="drop-zone" id="dropZone">
            <p>Drag & Drop files here or click to upload</p>
            <input type="file" id="fileInput" multiple style="display: none;">
        </div>
        <div class="progress-bar-container" id="progressBarContainer"><div class="progress-bar" id="uploadProgress">0%</div></div>

        <div class="tree-view" id="fileTree">Loading file tree...</div>
    </div>

    <script unSave="true">
        // Ensure TB (ToolBox Client-Side) is available


        // Ensure this script runs after the DOM is fully loaded,
        // and TB object is available.
        function initializeFileManager() {
        class FileManager {
            constructor() {
                this.dropZone = document.getElementById('dropZone');
                this.fileInput = document.getElementById('fileInput');
                this.fileTree = document.getElementById('fileTree');
                this.progressBarContainer = document.getElementById('progressBarContainer');
                this.uploadProgress = document.getElementById('uploadProgress');

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
            }

            async handleFiles(files) {
                for (const file of files) {
                    await this.uploadFile(file);
                }
            }

            async uploadFile(file) {
                this.progressBarContainer.style.display = 'block';
                this.uploadProgress.style.width = '0%';
                this.uploadProgress.textContent = '0%';
                const loaderId = TB.ui.Loader.show("Uploading " + file.name + "...");

                const chunkSize = 1 * 1024 * 1024; // 1MB
                const totalChunks = Math.ceil(file.size / chunkSize);
                let successfulChunks = 0;

                for (let i = 0; i < totalChunks; i++) {
                    const chunk = file.slice(i * chunkSize, (i + 1) * chunkSize);
                    const formData = new FormData();

                    // For HTTP fetch (browser native FormData handling):
                    // The 'name' attribute of formData.append('file', ...) is the field name.
                    // The Rust server will interpret this field name 'file' and its content.
                    formData.append('file', chunk, file.name); // The third arg to append (filename) sets Content-Disposition filename for this part

                    // These are the additional fields the Python function expects directly in form_data
                    formData.append('fileName', file.name);    // Overall intended filename
                    formData.append('chunkIndex', i.toString());        // Ensure string for form data
                    formData.append('totalChunks', totalChunks.toString());  // Ensure string for form data

                    try {
                        // For TB.api.request, if payload is FormData, it's passed directly for HTTP
                        // For Tauri, TB.api.request will transform it as discussed above.
                        const response = await TB.api.request(
                            'FileWidget',
                            'upload', // function_name for the Python endpoint
                            formData, // payload
                            'POST'    // method
                            // useTauri default is 'auto'
                        );

                        if (response && response.error && response.error !== "none") {
                            console.error('Upload chunk failed:', response.info?.help_text || response.error);
                            TB.ui.Toast.showError('Upload failed: ' + (response.info?.help_text || response.error));
                            this.progressBarContainer.style.display = 'none';
                            TB.ui.Loader.hide(loaderId);
                            return;
                        }
                        successfulChunks++;
                    } catch (error) {
                        console.error('Network error or TB.api.request issue during upload:', error);
                        TB.ui.Toast.showError('Upload failed: Network error or API call issue. ' + error.message);
                        this.progressBarContainer.style.display = 'none';
                        TB.ui.Loader.hide(loaderId);
                        return;
                    }
                    const progressVal = Math.round((successfulChunks / totalChunks) * 100);
                    this.uploadProgress.style.width = progressVal + '%';
                    this.uploadProgress.textContent = progressVal + '%';
                }

                TB.ui.Loader.hide(loaderId);
                if (successfulChunks === totalChunks) {
                    TB.ui.Toast.showSuccess(`File '${file.name}' uploaded successfully!`);
                    this.loadFileTree(); // Refresh tree after successful upload
                } else {
                    TB.ui.Toast.showError(`File '${file.name}' upload incomplete.`);
                }
                setTimeout(() => { this.progressBarContainer.style.display = 'none'; this.uploadProgress.style.width = '0%'; this.uploadProgress.textContent = '0%';}, 2000);
            }

            initLoadFileTree() {
                setTimeout(() => this.loadFileTree(), 100); // Short delay
            }

            async loadFileTree() {
                const loaderId = TB.ui.Loader.show("Loading files...");
                this.fileTree.innerHTML = '<i>Loading...</i>';
                try {
                    const response = await TB.api.request('FileWidget', 'files');
                    TB.ui.Loader.hide(loaderId);

                    if (response && response.result && response.result.data) {
                        this.renderFileTree(response.result.data);
                         if (Object.keys(response.result.data).length === 0) {
                             this.fileTree.innerHTML = "<p>No files or folders found.</p>";
                         }
                    } else if (response && response.error && response.error !== "none") {
                         this.fileTree.innerHTML = `<p style="color:red;">Error loading files: ${response.info?.help_text || response.error}</p>`;
                         TB.ui.Toast.showError(`Error loading files: ${response.info?.help_text || response.error}`);
                    } else {
                        this.fileTree.innerHTML = '<p>No files found or invalid response structure.</p>';
                        console.warn("File tree data not in expected format:", response);
                    }
                } catch (error) {
                    TB.ui.Loader.hide(loaderId);
                    this.fileTree.innerHTML = '<p style="color:red;">Failed to fetch file tree.</p>';
                    TB.ui.Toast.showError('Failed to fetch file tree: ' + error.message);
                    console.error("Error in loadFileTree:", error);
                }
            }

            renderFileTree(treeData) {
                this.responseCache = {}; // Clear cache
                this.fileTree.innerHTML = this.buildTreeHTML(treeData);
                if (!this.fileTree.innerHTML.trim()) { // Check if empty after build
                    this.fileTree.innerHTML = "<p>No files or folders.</p>";
                }
                this.addTreeEventListeners();
            }

            buildTreeHTML(node, currentPath = '') {
                let html = '';
                const entries = Object.entries(node).sort(([keyA, valA], [keyB, valB]) => {
                    const isDirA = typeof valA === 'object' && valA !== null && !valA.hasOwnProperty('content_base64'); // Heuristic for folder
                    const isDirB = typeof valB === 'object' && valB !== null && !valB.hasOwnProperty('content_base64');
                    if (isDirA !== isDirB) return isDirA ? -1 : 1;
                    return keyA.localeCompare(keyB);
                });

                for (const [name, content] of entries) {
                    const fullPathKey = currentPath ? `${currentPath}/${name}` : name;
                    // Check if content is an object and not null, and further check if it's a folder or file metadata
                    // A simple check: if content is a string, it's a file path. If an object, it's a folder (directory).
                    if (typeof content === 'object' && content !== null) { // It's a folder
                        html += `<div class="folder" data-folder-path="${fullPathKey}">📁 ${name}</div>`;
                        html += `<div class="folder-content">`;
                        html += this.buildTreeHTML(content, fullPathKey);
                        html += `</div>`;
                    } else { // It's a file, content is the actual path for download/share
                        const originalFilePath = String(content); // Server provides path as string
                        this.responseCache[fullPathKey] = originalFilePath;
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
                const iconMap = {'agent':'🤖','config':'🔧','json':'📋','pkl':'📦','txt':'📝','data':'💾','ipy':'🐍',
                'bin':'📀','sqlite3':'🗄️','vec':'📊','pickle':'🥒','html':'🌐','js':'📜','md':'📑','py':'🐍',
                'default':'📄', 'png':'🖼️', 'jpg':'🖼️', 'jpeg':'🖼️', 'gif':'🖼️', 'pdf':'📕',
                 'zip':'📦', 'csv':'📊', 'options': '⚙️'};
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
                            TB.ui.Toast.showError("Could not determine file path for download.");
                            console.error("Actual path not found for displayed path:", displayPath);
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
                            TB.ui.Toast.showError("Could not determine file path for sharing.");
                            console.error("Actual path not found for sharing:", displayPath);
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
                const loaderId = TB.ui.Loader.show("Preparing download...");
                try {
                    const response = await fetch(`/api/FileWidget/download?path=${encodeURIComponent(actualFilePathFromServer)}`);
                    TB.ui.Loader.hide(loaderId);
                    if (!response.ok) {
                        TB.ui.Toast.showError(`Error downloading file: ${response.statusText}`);
                        console.error("Download failed", response.status, await response.text());
                        return;
                    }

                    // Try to get filename from Content-Disposition header or fall back to path
                    let filename = actualFilePathFromServer.split('/').pop() || 'download';

                    const contentDisposition = response.headers.get('Content-Disposition');
                    if (contentDisposition) {
                        // RFC 6266 konforme Extraktion des Dateinamens
                        const filenameMatch = contentDisposition.match(/filename*=UTF-8''([^;]+)|filename="?([^"]+)"?/);

                        if (filenameMatch) {
                            filename = decodeURIComponent(filenameMatch[1] || filenameMatch[2] || filename).replace(/[/\\?%*:|"<>]/g, '-');
                        }
                    }

                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    TB.ui.Toast.showInfo("Download started.");
                } catch (error) {
                    TB.ui.Loader.hide(loaderId);
                    TB.ui.Toast.showError("Download error: " + error.message);
                    console.error("Download error:", error);
                }
            }

            async createShareLink(actualFilePathFromServer) {
                const loaderId = TB.ui.Loader.show("Creating share link...");
                try {
                    const response = await TB.api.request(
                        'FileWidget',
                        'create_share_link',
                        { file_path: actualFilePathFromServer, share_type: 'public' } // kwargs for python func
                    );
                    TB.ui.Loader.hide(loaderId);

                    if (response && response.result && response.result.data && response.result.data.share_link) {
                        const shareLink = response.result.data.share_link;

                        TB.ui.Modal.show({
                            title: 'Share Link Created',
                            content: `
                                <p>Your shareable link:</p>
                                <input type="text" id="shareLinkInputModal" value="${shareLink}" readonly style="width: 100%; padding: 8px; box-sizing: border-box; margin-bottom: 10px;">
                                <p><small>Anyone with this link can access the file.</small></p>
                            `,
                            buttons: [
                                {
                                    text: 'Copy Link',
                                    variant: 'primary',
                                    action: (modal) => {
                                        const input = document.getElementById('shareLinkInputModal');
                                        if(input) {
                                            input.select();
                                            input.setSelectionRange(0, 99999); // For mobile devices
                                            navigator.clipboard.writeText(input.value).then(() => {
                                                TB.ui.Toast.showSuccess('Share link copied to clipboard!');
                                            }).catch(err => {
                                                TB.ui.Toast.showError('Failed to copy link automatically.');
                                                console.error('Failed to copy: ', err);
                                            });
                                        }
                                    }
                                },
                                {
                                    text: 'Close',
                                    variant: 'secondary',
                                    action: (modal) => modal.hide()
                                }
                            ]
                        });
                    } else if (response && response.error && response.error !== "none") {
                        TB.ui.Toast.showError('Failed to create share link: ' + (response.info?.help_text || response.error));
                    } else {
                        TB.ui.Toast.showError('Could not retrieve share link from server response.');
                    }
                } catch (error) {
                    TB.ui.Loader.hide(loaderId);
                    console.error("Error creating share link:", error);
                    TB.ui.Toast.showError('Error creating share link: ' + error.message);
                }
            }
        }

            new FileManager();
        }

        if (window.TB?.events) {
            if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
                 initializeFileManager();
            } else {
                window.TB.events.on('tbjs:initialized', initializeFileManager, { once: true });
            }
        } else {
            // Fallback if TB is not even an object yet, very early load
            document.addEventListener('tbjs:initialized', initializeFileManager, { once: true }); // Custom event dispatch from TB.init
        }

    </script>
    """


# --- API Endpoints using @export ---
@export(mod_name=MOD_NAME, api=True, version=VERSION, name="ui", api_methods=['GET'])
async def get_main_ui(self) -> Result:
    """Serves the main HTML UI for the FileWidget."""
    html_content = get_template_content()
    return Result.html(data=html_content)


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="upload", api_methods=['POST'], request_as_kwarg=True)
async def handle_upload(self, request: RequestData, form_data: dict[str, Any] | None = None) -> Result:
    """
    Handles file uploads. Expects chunked data via form_data kwarg from Rust server.
    'form_data' structure (from Rust's parsing of multipart) after client sends FormData with fields:
    'file' (the blob), 'fileName', 'chunkIndex', 'totalChunks'.

    Expected `form_data` in this Python function:
    {
        "file": {  // This 'file' key is the NAME of the form field that held the file blob
            "filename": "original_file_name_for_this_chunk.txt", // from Content-Disposition of the 'file' field part
            "content_type": "mime/type_of_chunk",
            "content_base64": "BASE64_ENCODED_CHUNK_CONTENT"
        },
        "fileName": "overall_final_filename.txt", // From a separate form field named 'fileName'
        "chunkIndex": "0",                        // From a separate form field named 'chunkIndex'
        "totalChunks": "5"                        // From a separate form field named 'totalChunks'
    }
    """
    self.app.logger.debug(
        f"FileWidget: handle_upload called. Received form_data keys: {list(form_data.keys()) if form_data else 'None'}"
    )
    self.app.logger.debug(f"FileWidget: handle_upload called. Received form_data: {request.to_dict()}")
    # self.app.logger.debug(f"Full form_data: {form_data}") # For deeper debugging if needed

    if not form_data:
        return Result.default_user_error(info="No form data received for upload.", exec_code=400)

    try:
        storage = await self.get_blob_storage(request)

        # Extract data from form_data (populated by Rust server from multipart)
        file_field_data = form_data.get('file')  # This is the dict from UploadedFile struct
        # The 'file_field_data.get('filename')' is the name of the chunk part,
        # which the JS client sets to be the same as the original file's name.
        # This is fine for FileUploadHandler.save_file's chunk_info.filename if total_chunks > 1,
        # as it will be used to create temporary part files like "original_file_name.txt.part0".

        overall_filename_from_form = form_data.get('fileName') # This is the target filename for the assembled file.
        chunk_index_str = form_data.get('chunkIndex')
        total_chunks_str = form_data.get('totalChunks')

        if not all([
            file_field_data, isinstance(file_field_data, dict),
            overall_filename_from_form,
            chunk_index_str is not None, # Check for presence, not just truthiness (0 is valid)
            total_chunks_str is not None # Check for presence
        ]):
            missing = []
            if not file_field_data or not isinstance(file_field_data, dict): missing.append("'file' object field")
            if not overall_filename_from_form: missing.append("'fileName' field")
            if chunk_index_str is None: missing.append("'chunkIndex' field")
            if total_chunks_str is None: missing.append("'totalChunks' field")

            self.app.logger.error(
                f"Missing critical form data fields for upload: {missing}. Received form_data: {form_data}")
            return Result.default_user_error(info=f"Incomplete upload data. Missing: {', '.join(missing)}",
                                             exec_code=400)

        content_base64 = file_field_data.get('content_base64')
        if not content_base64:
            return Result.default_user_error(info="File content (base64) not found in 'file' field data.",
                                             exec_code=400)

        try:
            content_bytes = base64.b64decode(content_base64)
        except base64.binascii.Error as b64_error:
            self.app.logger.error(f"Base64 decoding failed for upload: {b64_error}")
            return Result.default_user_error(info="Invalid file content encoding.", exec_code=400)

        try:
            chunk_index = int(chunk_index_str)
            total_chunks = int(total_chunks_str)
        except ValueError:
            return Result.default_user_error(info="Invalid chunk index or total chunks value. Must be integers.", exec_code=400)

        # Use the 'overall_filename_from_form' for the ChunkInfo.filename,
        # as this is the intended final name in blob storage.
        # FileUploadHandler will use Path(this_name).name to ensure it's just a filename.
        chunk_info_to_save = ChunkInfo(
            filename=overall_filename_from_form, # THIS IS THE KEY CHANGE FOR CONSISTENCY
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=content_bytes
        )

        self.app.logger.info(
            f"Processing chunk {chunk_index + 1}/{total_chunks} for final file '{overall_filename_from_form}'. " # Log the intended final name
            f"Size: {len(content_bytes)} bytes."
        )

        saved_blob_path = self.upload_handler.save_file(chunk_info_to_save, storage) # saved_blob_path will be Path(overall_filename_from_form).name

        msg = f"Chunk {chunk_index + 1}/{total_chunks} for '{saved_blob_path}' saved."
        if chunk_info_to_save.chunk_index == chunk_info_to_save.total_chunks - 1:
            # Check if fully assembled
            # The 'safe_base_filename' in FileUploadHandler is derived from ChunkInfo.filename,
            # which we've now set to 'overall_filename_from_form'.
            # So, this check should work correctly.
            safe_base_filename_for_check = "".join(
                c if c.isalnum() or c in ('.', '_', '-') else '_' for c in Path(overall_filename_from_form).name)

            # A slight delay might be needed if file system operations are not instantly consistent across threads/processes
            # For now, assume direct check is okay.
            # await asyncio.sleep(0.1) # Optional small delay if race conditions are suspected with file system

            if self.upload_handler._all_chunks_received(safe_base_filename_for_check, total_chunks):
                msg = f"File '{saved_blob_path}' upload complete and assembled."
                self.app.logger.info(msg)
            else:
                msg = f"Final chunk for '{saved_blob_path}' saved, but assembly check failed or is pending."
                self.app.logger.warning(msg + f" (Could not verify all chunks for '{safe_base_filename_for_check}' immediately after final one)")


        return Result.ok(data={"message": msg, "path": saved_blob_path}) # Return the blob-relative path

    except ValueError as e:
        self.app.logger.error(f"Upload processing error: {e}", exc_info=True)
        return Result.default_user_error(info=f"Upload error: {str(e)}",
                                         exec_code=400 if "authentication" in str(e).lower() else 400)
    except Exception as e:
        self.app.logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
        return Result.default_internal_error(info="An unexpected error occurred during upload.")


async def _prepare_file_response(self, storage: BlobStorage, blob_path: str, row=False) -> Result:
    try:
        # Basic sanitization for blob_path. BlobFile should handle sandboxing.
        # Ensure blob_path is relative and doesn't try to escape.
        # Path(blob_path).is_absolute() or ".." in blob_path are checks one might do.
        # For BlobStorage, paths are relative to its root. Path(blob_path).name could be used
        # if only files at root are allowed, but full relative paths are generally fine.
        if ".." in blob_path or Path(blob_path).is_absolute():
            self.app.logger.warning(f"Attempt to access potentially unsafe path: {blob_path}")
            return Result.default_user_error(info="Invalid file path.", exec_code=400)

        if not BlobFile(blob_path, storage=storage).exists():
            self.app.logger.warning(f"File not found in BlobStorage: {blob_path}")
            return Result.default_user_error(info="File not found.", exec_code=404)

        filename = Path(blob_path).name
        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = 'application/octet-stream'
        with BlobFile(blob_path, 'r', storage=storage) as bf:
            data = bf.read()
        file_size = len(data)

        async def file_streamer() -> AsyncGenerator[bytes, None]:
            with BlobFile(blob_path, 'r', storage=storage) as bf:
                # Stream in chunks for large files
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    data_chunk = bf.read(chunk_size)
                    if not data_chunk:
                        break
                    yield data_chunk
        if row:
            return Result.file(data, filename)

        self.app.logger.info(f"Preparing file download '{blob_path}' ({content_type}, {file_size} bytes).")

        # Convert to base64 for embedding in HTML
        data_base64 = base64.b64encode(data).decode('utf-8')

        # Create HTML that automatically downloads the file with correct filename
        download_html = f"""<div>
    <title>Downloading {filename}</title>
    <div style="padding: 50px; font-family: Arial, sans-serif;">
        <h2>Downloading {filename}</h2>
        <section>
        <p>Your download should start automatically...</p>
        <p><a id="download-link" href="#" download="{filename}">Click here if download doesn't start</a></p>
        </section>
    </div>
    <script unSave="true">
    if (window.TB) {{
        TB.ui.Toast.showInfo("Download started.");

        // Convert base64 to blob and trigger download
        const base64Data = "{data_base64}";
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {{
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }}
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], {{type: "{content_type}"}});

        const url = URL.createObjectURL(blob);
        const link = document.getElementById('download-link');
        link.href = url;
        link.download = "{filename}";

         // Get the current URL and its query parameters
          const currentUrl = new URL(window.location.href);

          // Add or set the query parameter `row=true`
          currentUrl.searchParams.set('row', 'true');

          // Set this URL to the download link
          const downloadLink = document.getElementById('download-link');
          downloadLink.href = currentUrl.toString();

        // Auto-trigger download
        setTimeout(() => {{
            link.click();
            // Clean up
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }}, 100);
    }}
    </script>
</div>"""

        result = Result.html(data=download_html)
        return result
    except FileNotFoundError:
        self.app.logger.warning(f"Download attempt for non-existent file (caught late): {blob_path}")
        return Result.default_user_error(info="File not found.", exec_code=404)
    except Exception as e:
        self.app.logger.error(f"Error processing download for {blob_path}: {e}", exc_info=True)
        self.app.debug_rains(e)
        return Result.default_internal_error(info="Error processing download.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="download", row=True, api_methods=['GET'], request_as_kwarg=True, test=False)
async def handle_download(self, request: RequestData, **kwargs) -> Result:  # Removed path kwarg, always use query_params
    blob_path = request.query_params.get('path')
    if not blob_path:
        return Result.default_user_error(info="File path parameter 'path' is missing.", exec_code=400)

    try:
        # User's own storage. get_blob_storage handles auth check.
        storage = await self.get_blob_storage(request)
        self.app.logger.info(f"User download request for: {blob_path} from their storage.")
        result = await _prepare_file_response(self, storage, blob_path, row=True)
        return result
    except ValueError as e:  # Raised by get_blob_storage if user not authenticated
        self.app.logger.warning(f"Auth error during download for path {blob_path}: {e}")
        return Result.default_user_error(info=str(e), exec_code=401)  # 401 Unauthorized
    except Exception as e:
        self.app.logger.error(f"Unexpected error in handle_download for {blob_path}: {e}", exc_info=True)
        return Result.default_internal_error(info="Failed to process download request.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="files", api_methods=['GET'], request_as_kwarg=True)
async def get_file_tree(self, request: RequestData) -> Result:
    try:
        storage = await self.get_blob_storage(request)  # Handles auth
        tree: dict[str, Any] = {}
        all_paths = []

        # Adapt to BlobStorage's way of listing files.
        # Assuming BlobStorage might have a more direct way or we fall back to iterating.
        if hasattr(storage, 'list_files'):  # Ideal: storage.list_files(recursive=True) -> List[str]
            all_paths = storage.list_files(recursive=True)
        elif hasattr(storage, '_get_all_blob_ids_and_paths'):  # Method from original thought process
            all_paths = storage._get_all_blob_ids_and_paths()
        elif hasattr(storage, '_get_all_blob_ids'):  # Fallback if only opaque IDs are available
            blob_ids = storage._get_all_blob_ids()
            # This implies blob_id is the path or can be resolved to a path.
            # The original code had complex pickle loading here which is highly specific.
            # Assuming blob_id itself is the path string for simplicity.
            all_paths = [str(bid) for bid in blob_ids]  # Filter out potentially invalid ids
        elif storage.storage_root and Path(storage.storage_root).exists():  # Filesystem-based fallback
            self.app.logger.debug(f"Falling back to filesystem scan for BlobStorage at {storage.storage_root}")
            for item in Path(storage.storage_root).rglob('*'):
                if item.is_file():
                    all_paths.append(str(item.relative_to(storage.storage_root)))
        else:
            self.app.logger.warning("BlobStorage does not support standard listing methods, and no fallback possible.")
            return Result.ok(data={})  # Return empty if listing is not possible

        for file_path_str in sorted(all_paths):  # Sort paths for consistent tree structure
            path_parts = Path(file_path_str).parts  # Use Path parts for OS-agnostic splitting
            current_level = tree
            for i, part in enumerate(path_parts):
                if not part: continue
                if i == len(path_parts) - 1:  # It's a file
                    current_level[part] = file_path_str  # Store the full original path as value
                else:  # It's a folder
                    current_level = current_level.setdefault(part, {})
                    if not isinstance(current_level, dict):  # Conflict: file with same name as folder part
                        self.app.logger.warning(
                            f"File/folder name conflict for '{part}' in path '{file_path_str}'. Overwriting with folder.")
                        # This case should ideally be prevented by design or handled by BlobStorage's structure.
                        # For now, if a file exists where a folder should be, we might have an issue.
                        # Simplistic approach: If a non-dict is found, create new dict, potentially losing the file.
                        # A better approach: append a special marker or error.
                        # current_level[part] = {} # Force it to be a dict for further path parts
                        # current_level = current_level[part]
                        # Let's log and skip this problematic path to avoid data corruption in tree.
                        self.app.logger.error(
                            f"Path conflict: {file_path_str} - segment {part} is a file but needs to be a directory.")
                        break  # Stop processing this conflicted path
            else:  # Inner loop completed without break
                continue
            break  # Outer loop break if inner loop broke

        self.app.logger.debug(f"File tree for user: {json.dumps(tree, indent=2)}")
        return Result.json(data=tree)
    except ValueError as e:
        self.app.logger.warning(f"Auth error during file tree access: {e}")
        return Result.default_user_error(info=str(e), exec_code=401)
    except Exception as e:
        self.app.logger.error(f"Unexpected error in get_file_tree: {e}", exc_info=True)
        return Result.default_internal_error(info="Failed to retrieve file list.")


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="create_share_link_test", api_methods=['GET', 'POST'],
        request_as_kwarg=True)
async def create_share_link_test(self):

    return await create_share_link(self, RequestData.moc(), "init.config")


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="create_share_link", api_methods=['GET', 'POST'],
        request_as_kwarg=True)
async def create_share_link(self, request: RequestData, file_path: str | None = None,
                            share_type: str | None = 'public') -> Result:
    user_uid = await self._get_user_uid_from_request(request)
    if not user_uid:  # Should be caught by get_blob_storage called next, but explicit check is good.
        return Result.default_user_error(info="Authentication required to create share links.", exec_code=401)

    # Determine file_path from query_params (GET) or JSON body (POST)
    # The function signature allows direct passing too, useful for internal calls.
    if request.method == 'POST':
        json_body = request.form_data or {}
        file_path = file_path or json_body.get('file_path')
        share_type = share_type or json_body.get('share_type', 'public')
    elif request.method == 'GET':
        file_path = file_path or request.query_params.get('file_path')
        share_type = share_type or request.query_params.get('share_type', 'public')

    if not file_path:
        return Result.default_user_error(info="Parameter 'file_path' is required.", exec_code=400)

    try:
        user_storage: BlobStorage = await self.get_blob_storage(request)  # Validates user auth implicitly
        if not BlobFile(file_path, storage=user_storage).exists():
            return Result.default_user_error(info=f"File not found in your storage: {file_path}", exec_code=404)
    except ValueError as e:  # From get_blob_storage if auth fails
        return Result.default_user_error(info=str(e), exec_code=401)

    share_id = self._generate_share_id()
    self.shares[share_id] = {
        "owner_uid": user_uid,
        "file_path": file_path,
        "created_at": time.time(),
        "share_type": share_type
    }
    self._save_shares()

    # Construct the full share link URL. APP_BASE_URL should be set in your app's environment.
    # Include the filename in the URL so the browser can use it for download
    from pathlib import Path
    filename = Path(file_path).name
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")
    share_access_path = f"/api/{MOD_NAME}/open_shared?share_id={share_id}&filename={filename}"  # Include filename
    full_share_link = str(base_url).rstrip('/') + share_access_path

    self.app.logger.info(
        f"Share link created by UID {user_uid} for path '{file_path}': ID {share_id}, Link: {full_share_link}")
    return Result.ok(data={"share_id": share_id, "share_link": full_share_link})


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="open_shared", api_methods=['GET'],
        request_as_kwarg=True, level=-1, row=True)
async def access_shared_file(self, request: RequestData, share_id: str, filename: str = None, row=None) -> Result:  # share_id from query params
    """
    Accesses a shared file via its share_id.
    The URL for this would be like /api/FileWidget/shared/{share_id_value}
    The 'share_id: str' in signature implies ToolBoxV2 extracts it from path.
    """
    if not share_id:
        return Result.html(data="Share ID is missing in path.", status=302)

    share_info = self.shares.get(share_id) if self.shares is not None else None
    if not share_info:
        return Result.html(data="Share link is invalid or has expired.", status=404)

    owner_uid = share_info["owner_uid"]
    file_path_in_owner_storage = share_info["file_path"]

    try:
        # Get BlobStorage for the owner, not the current request's user (if any)
        owner_storage = await self.get_blob_storage(
            owner_uid_override=owner_uid)  # Crucially, pass request=None if not needed
        self.app.logger.info(
            f"Accessing shared file via link {share_id}: owner {owner_uid}, path {file_path_in_owner_storage}")
        result = await _prepare_file_response(self, owner_storage, file_path_in_owner_storage, row=row is not None)
        if result.is_error():
            self.app.logger.error(f"Error preparing shared file response for {share_id}: {result.info.help_text}")
            return Result.html(data=f"Failed to prepare shared file for download. {result.info.help_text} {result.result.data_info}")
        return result
    except ValueError as e:  # From get_blob_storage if owner_uid is invalid for some reason
        self.app.logger.error(f"Error getting owner's storage for shared file {share_id} (owner {owner_uid}): {e}",
                              exc_info=True)
        return Result.html(data="Could not access owner's storage for shared file.")
    except Exception as e:
        self.app.logger.error(
            f"Error accessing shared file {share_id} (owner {owner_uid}, path {file_path_in_owner_storage}): {e}",
            exc_info=True)
        return Result.html(data="Could not retrieve shared file.")
