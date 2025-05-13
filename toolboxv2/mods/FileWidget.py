import mimetypes
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from starlette.responses import Response

from toolboxv2 import MainTool
from toolboxv2.utils.extras import BaseWidget
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage
from toolboxv2.utils.system.session import RequestSession


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

    def _extract_boundary(self) -> str:
        # Erste Zeile enthält die Boundary
        first_line = self.body.split(b'\r\n')[0]
        return first_line.decode('utf-8')

    def _parse_content_disposition(self, headers: str) -> dict:
        result = {}
        for header in headers.split('\r\n'):
            if header.startswith('Content-Disposition'):
                # Parse name und filename
                matches = re.findall(r'(\w+)="([^"]+)"', header)
                for key, value in matches:
                    result[key] = value
        return result

    def parse(self) -> ChunkInfo:
        # Split an Boundary
        parts = self.body.split(bytes(self.boundary, 'utf-8'))

        file_content = None
        filename = None
        chunk_index = None
        total_chunks = None

        for part in parts:
            if not part.strip():
                continue

            # Trennen von Headers und Content
            try:
                headers, content = part.split(b'\r\n\r\n', 1)
                headers = headers.decode('utf-8')

                # Parse Content-Disposition
                disposition = self._parse_content_disposition(headers)

                if disposition.get('name') == 'file':
                    file_content = content.rsplit(b'\r\n', 1)[0]
                elif disposition.get('name') == 'fileName':
                    filename = content.strip(b'\r\n').decode('utf-8')
                elif disposition.get('name') == 'chunkIndex':
                    chunk_index = int(content.strip(b'\r\n'))
                elif disposition.get('name') == 'totalChunks':
                    total_chunks = int(content.strip(b'\r\n'))
            except:
                continue

        return ChunkInfo(
            filename=filename,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=file_content
        )


class FileUploadHandler:
    def __init__(self, upload_dir: str = 'uploads'):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def save_file(self, chunk_info: ChunkInfo, storage=None) -> str:
        """Speichert die Datei oder Chunk"""
        if chunk_info.total_chunks == 1:
            # Komplette Datei speichern
            filepath = os.path.join(self.upload_dir, chunk_info.filename)
            with BlobFile(filepath, 'w', storage=storage) as f:
                f.write(chunk_info.content)
        else:
            # Chunk speichern
            chunk_path = os.path.join(
                self.upload_dir,
                f"{chunk_info.filename}.part{chunk_info.chunk_index}"
            )
            with open(chunk_path, 'wb') as f:
                f.write(chunk_info.content)

            # Wenn alle Chunks da sind, zusammenfügen
            if self._all_chunks_received(chunk_info):
                self._merge_chunks(chunk_info, storage=storage)
                self._cleanup_chunks(chunk_info)

        return os.path.join(self.upload_dir, chunk_info.filename)

    def _all_chunks_received(self, chunk_info: ChunkInfo) -> bool:
        """Prüft ob alle Chunks empfangen wurden"""
        if chunk_info.total_chunks is None:
            return False

        for i in range(chunk_info.total_chunks):
            chunk_path = os.path.join(
                self.upload_dir,
                f"{chunk_info.filename}.part{i}"
            )
            if not os.path.exists(chunk_path):
                return False
        return True

    def _merge_chunks(self, chunk_info: ChunkInfo, storage=None):
        """Fügt alle Chunks zusammen"""
        filepath = os.path.join(self.upload_dir, chunk_info.filename)
        print("filepath", filepath)
        with BlobFile(filepath, 'w', storage=storage) as outfile:
            for i in range(chunk_info.total_chunks):
                chunk_path = os.path.join(
                    self.upload_dir,
                    f"{chunk_info.filename}.part{i}"
                )
                with open(chunk_path, 'rb') as chunk:
                    outfile.write(chunk.read())

    def _cleanup_chunks(self, chunk_info: ChunkInfo):
        """Löscht temporäre Chunk-Dateien"""
        for i in range(chunk_info.total_chunks):
            chunk_path = os.path.join(
                self.upload_dir,
                f"{chunk_info.filename}.part{i}"
            )
            if os.path.exists(chunk_path):
                os.remove(chunk_path)


Name = "FileWidget"
version = "0.0.1"


class FileWidget(MainTool, BaseWidget):
    def __init__(self, app=None):
        self.name = "FileWidget"
        self.color = "WHITE"
        self.tools = {'name': self.name}
        self.version = "1.0.0"
        MainTool.__init__(self,
                          load=self.on_start,
                          v=self.version,
                          name=self.name,
                          color=self.color,
                          on_exit=self.on_exit)

        BaseWidget.__init__(self, name=self.name)
        self.register(self.app, self.get_widget, self.version)
        self.register(self.app, self.handle_upload, self.version, name="upload")
        self.register(self.app, self.handle_download, self.version, name="download", row=True)
        self.register(self.app, self.get_file_tree, self.version, name="files")

        self.blob_storage = {}

    async def get_blob_storage(self, request):
        user = await self.get_user_from_request(self.app, request)
        if user.name == "":
            return BlobStorage(self.app.data_dir + '/public', 0)
        if user.name == "root":
            return BlobStorage()
        if user.name not in self.blob_storage:
            self.blob_storage[user.name] = BlobStorage(
                self.app.data_dir + '/storages/' + user.uid)
        return self.blob_storage[user.name]

    def main(self, request):
        w_id = self.get_s_id(request)
        if w_id.is_error():
            return w_id
        self.asset_loder(self.app, "main", self.hash_wrapper(w_id.get()), template=self.get_template())

    def get_template(self):
        return """
        <title>File Manager</title>
        <style>
        .tree-view {
            font-family: monospace;
            margin: 10px 0;
            border: 1px solid #ddd;
            padding: 10px;
            max-height: 600px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .folder-group {
            font-weight: bold;
            color: #2c3e50;
            padding: 5px;
            margin-top: 10px;
            background: #edf2f7;
            border-radius: 4px;
            cursor: pointer;
        }

        .group-content {
            margin-left: 20px;
            border-left: 2px solid #e2e8f0;
            padding-left: 10px;
        }

        .folder {
            cursor: pointer;
            padding: 2px 5px;
            margin: 2px 0;
            color: #4a5568;
        }

        .folder:hover {
            background: #edf2f7;
            border-radius: 4px;
        }

        .file {
            padding: 2px 5px;
            margin: 2px 0;
            cursor: pointer;
            color: #718096;
            transition: background-color 0.2s;
        }

        .file:hover {
            background: #edf2f7;
            border-radius: 4px;
            color: #2d3748;
        }

        .folder-content {
            margin-left: 20px;
            border-left: 1px solid #e2e8f0;
            padding-left: 10px;
            display: none;
        }


        .folder-content.open {
            display: block;
        }

        .folder::before {
            content: '▶'; /* Geschlossener Pfeil */
            display: inline-block;
            margin-right: 5px;
            transition: transform 0.2s;
        }

        .folder.open::before {
            transform: rotate(90deg); /* Pfeil dreht sich beim Öffnen */
        }

        .group-content {
            display: none; /* Standardmäßig eingeklappt */
        }

        .group-content.open {
            display: block;
        }

        .folder-group::before {
            content: '▶';
            display: inline-block;
            margin-right: 5px;
            transition: transform 0.2s;
        }

        .folder-group.open::before {
            transform: rotate(90deg);
        }

        .drop-zone {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            cursor: pointer;
        }

        .drop-zone.dragover {
            background-color: #e1e1e1;
            border-color: #999;
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress {
            width: 0%;
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease-in-out;
        }
        </style>

        <div class="file-container">
            <h2>File Manager</h2>

            <div class="drop-zone" id="dropZone">
                <p>Drag & Drop files here or click to upload</p>
                <input type="file" id="fileInput" multiple style="display: none;">
            </div>

            <div class="progress-bar" style="display: none;">
                <div class="progress" id="uploadProgress"></div>
            </div>

            <div class="tree-view" id="fileTree"></div>
        </div>

        <script unSave="true">
            class FileManager {
                constructor() {
                    this.dropZone = document.getElementById('dropZone');
                    this.fileInput = document.getElementById('fileInput');
                    this.fileTree = document.getElementById('fileTree');
                    this.progressBar = document.querySelector('.progress-bar');
                    this.progress = document.getElementById('uploadProgress');
                    this.response = {};

                    this.initEventListeners();
                    this.initLoadFileTree();
                }

                initEventListeners() {
                    this.dropZone.addEventListener('click', () => this.fileInput.click());
                    this.fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));

                    this.dropZone.addEventListener('dragover', (e) => {
                        e.preventDefault();
                        this.dropZone.classList.add('dragover');
                    });

                    this.dropZone.addEventListener('dragleave', () => {
                        this.dropZone.classList.remove('dragover');
                    });

                    this.dropZone.addEventListener('drop', (e) => {
                        e.preventDefault();
                        this.dropZone.classList.remove('dragover');
                        this.handleFiles(e.dataTransfer.files);
                    });
                }

                async handleFiles(files) {
                    for (const file of files) {
                        await this.uploadFile(file);
                    }
                    this.loadFileTree();
                }

                async uploadFile(file) {
                    this.progressBar.style.display = 'block';

                    const chunkSize = 1024 * 1024; // 1MB chunks
                    const totalChunks = Math.ceil(file.size / chunkSize);

                    for (let i = 0; i < totalChunks; i++) {
                        const chunk = file.slice(i * chunkSize, (i + 1) * chunkSize);
                        const formData = new FormData();
                        formData.append('file', chunk);
                        formData.append('fileName', file.name);
                        formData.append('chunkIndex', i);
                        formData.append('totalChunks', totalChunks);

                        await fetch('/api/FileWidget/upload', {
                            method: 'POST',
                            body: formData
                        });

                        const progress = ((i + 1) / totalChunks) * 100;
                        this.progress.style.width = progress + '%';
                    }

                    setTimeout(() => {
                        this.progressBar.style.display = 'none';
                        this.progress.style.width = '0%';
                    }, 1000);
                }

                initLoadFileTree() {
                    setTimeout(async () => {
                        await this.loadFileTree();
                    }, 1000);
                }
                async loadFileTree() {
                    const response = await fetch('/api/FileWidget/files');
                    const files = await response.json();
                    this.renderFileTree(files);
                }

                renderFileTree(files) {
                    this.fileTree.innerHTML = this.buildTreeHTML(files);
                    this.addTreeEventListeners();
                }

                buildTreeHTML(response, level = 0) {
                    // Überprüfe auf API-Antwortstruktur und extrahiere die relevanten Daten
                    if (typeof(response) == "string"){
                        response = JSON.parse(response);
                    }
                    console.log("buildTreeHTML", response, response.result)
                    if (response.result && response.result.data) {
                        return this.buildTreeHTML(response.result.data, level);
                    }
                    if (response['result'] && response['result']['data']) {
                        return this.buildTreeHTML(response['result']['data'], level);
                    }
                    this.response = Object.assign({}, this.response, response);
                    // Wenn es ein einfaches Key-Value Paar ist
                    if (typeof response === 'object' && !Array.isArray(response)) {
                        let html = '';
                        const sorted = Object.entries(response).sort(([keyA], [keyB]) => {
                            const extA = keyA.split('.').pop() || '';
                            const extB = keyB.split('.').pop() || '';
                            if (extA === extB) {
                                return keyA.localeCompare(keyB);
                            }
                            return extA.localeCompare(extB);
                        });

                        let currentGroup = '';
                        let indent =  '';
                        for (const [key, value] of sorted) {
                            indent = '    '.repeat(level);
                            const fileExt = key.split('.').pop() || '';

                            if (fileExt !== currentGroup) {
                                currentGroup = fileExt;
                                if (level === 0) {
                                    html += indent + '<div class="folder-group">📁 ' + currentGroup.toUpperCase() + ' Files</div>';
                                    html += indent + '<div class="group-content" style="margin-left: 20px">';
                                }
                            }

                            const icon = this.getFileIcon(key);

                            if (typeof value === 'object' && value !== null) {
                                html += indent + '<div class="folder" data-folder="' + key + '">📁 ' + key + '</div>';
                                html += indent + '<div class="folder-content" style="margin-left: 20px">';
                                html += this.buildTreeHTML(value, level + 1);
                                html += indent + '</div>';
                            } else {
                                html += indent + '<div class="file" data-path="' + key + '">' + icon + ' ' + key + '</div>';
                            }

                            const nextEntry = sorted[sorted.indexOf([key, value]) + 1];
                            if (level === 0 && nextEntry) {
                                const nextExt = nextEntry[0].split('.').pop() || '';
                                if (nextExt !== currentGroup) {
                                    html += indent + '</div>';
                                }
                            }
                        }

                        if (level === 0 && currentGroup) {
                            html += indent + '</div>';
                        }

                        return html;
                    }

                    if (typeof response === 'string') {
                        const icon = this.getFileIcon(response);
                        return '<div class="file" data-path="' + response + '">' + icon + ' ' + response + '</div>';
                    }

                    return '';
                }

                getFileIcon(filename) {
                    const ext = filename.split('.').pop()?.toLowerCase();
                    const iconMap = {
                        'agent': '🤖',
                        'json': '📋',
                        'pkl': '📦',
                        'txt': '📝',
                        'data': '💾',
                        'ipy': '🐍',
                        'bin': '📀',
                        'sqlite3': '🗄️',
                        'vec': '📊',
                        'pickle': '🥒',
                        'html': '🌐',
                        'javascript': '📜',
                        'markdown': '📑',
                        'python': '🐍',
                        'default': '📄'
                    };
                    return iconMap[ext] || iconMap['default'];
                }

                addTreeEventListeners() {
                    document.querySelectorAll('.file').forEach(file => {
                        file.addEventListener('click', () => this.downloadFile(file.dataset.path));
                    });

                    // Neue Event-Listener für Ordner
                    document.querySelectorAll('.folder').forEach(folder => {
                        folder.addEventListener('click', (e) => {
                            e.stopPropagation();
                            folder.classList.toggle('open');
                            const content = folder.nextElementSibling;
                            if (content && content.classList.contains('folder-content')) {
                                content.classList.toggle('open');
                            }
                        });
                    });

                    // Event-Listener für Gruppen
                    document.querySelectorAll('.folder-group').forEach(group => {
                        group.addEventListener('click', (e) => {
                            e.stopPropagation();
                            group.classList.toggle('open');
                            const content = group.nextElementSibling;
                            if (content && content.classList.contains('group-content')) {
                                content.classList.toggle('open');
                            }
                        });
                    });
                }
                async downloadFile(path) {
                    if (this.response){
                        path = this.response[path]
                    }
                    const response = await fetch(`/api/FileWidget/download?path=`+encodeURIComponent(path));
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = path.split('/').pop();
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                }
            }

            new FileManager();
        </script>
        """

    async def handle_upload(self, request: RequestSession):

        if request is None:
            return None

        body = request.body
        storage = await self.get_blob_storage(request)

        parser = MultipartParser(body)
        chunk_info = parser.parse()

        handler = FileUploadHandler()
        saved_path = handler.save_file(chunk_info, storage)
        return saved_path

        ## Erstellen oder aktualisieren der BlobFile
#
        #with BlobFile('userData/'+file.filename, 'w', storage=storage) as bf:
        #    while contents := file.file.read(1024 * 1024):
        #        bf.write(contents)


    async def handle_download(self, request, path):
        """
            Handle file downloads for BlobFile using Starlette response.

            Args:
                request: The Starlette request object
                path: The blob file path to download

            Returns:
                Starlette Response with file data and appropriate headers
            """
        try:
            # Remove leading slash if present for consistency with BlobFile
            if path.startswith('/'):
                path = path[1:]

            # Get the filename from the path
            filename = Path(path).name

            # Detect content type based on file extension
            content_type, _ = mimetypes.guess_type(filename)
            if content_type is None:
                content_type = 'application/octet-stream'
            storage = await self.get_blob_storage(request)
            # Read the file data
            with BlobFile(path, 'r', storage=storage) as bf:
                data = bf.read()

            # Create headers for the response
            headers = {
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(len(data)),
                'Content-Type': content_type
            }

            return Response(
                content=data,
                headers=headers,
                media_type=content_type
            )

        except FileNotFoundError:
            return Response(
                content="File not found",
                status_code=404
            )
        except Exception as e:
            return Response(
                content=f"Error processing download: {str(e)}",
                status_code=500
            )

    async def get_file_tree(self, request):

        def flatten_dict(d, parent_key='', sep='.'):
            items = {}
            for k, v in d.items():
                # Erstelle einen neuen Schlüssel durch Anhängen des aktuellen Schlüssels
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                # Wenn der Wert wieder ein Dictionary ist, rufe die Funktion rekursiv auf
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        # Implementierung der Verzeichnisstruktur
        tree = {}
        storage = await self.get_blob_storage(request)
        blob_ids = storage._get_all_blob_ids()
        folder_list = []
        for blob_id in blob_ids:
            blob_data = pickle.loads(storage.read_blob(blob_id))
            folder_list.extend(flatten_dict(blob_data, blob_id, '/').keys())

        for folder in folder_list:
            path_parts = folder.split('/')
            current = tree
            for part in path_parts[:-1]:
                if not part:
                    continue
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[path_parts[-1]] = folder

        return tree

    def on_start(self):
        self.register2reload(self.main)
        # API-Routen registrieren

    def on_exit(self):
        pass

    async def get_widget(self, request, **kwargs):
        w_id = self.get_s_id(request)
        if w_id.is_error():
            return w_id
        self.reload_guard(self.on_start)
        return self.load_widget(self.app, request, "main", self.hash_wrapper(w_id.get()))


Tools = FileWidget
