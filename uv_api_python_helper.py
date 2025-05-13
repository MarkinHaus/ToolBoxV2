import os
import platform
import subprocess
import tarfile
import urllib.request


def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)


def python_exists(target_dir, version):
    python_exe = os.path.join(target_dir, "python.exe" if platform.system() == "Windows" else "bin/python3")
    if os.path.exists(python_exe):
        try:
            output = subprocess.check_output([python_exe, "--version"], stderr=subprocess.STDOUT).decode()
            if version in output:
                print(f"Python {version} already installed.")
                return python_exe
        except Exception as e:
            print("ERROR", e)
    return None


def install_python(target_dir, version="3.12.9"):
    python_exe = python_exists(target_dir, version)
    if python_exe:
        return python_exe

    system = platform.system()
    if system == "Windows":
        python_url = f"https://www.python.org/ftp/python/{version}/python-{version}-amd64.exe"
        installer_path = os.path.join(target_dir, "python-installer.exe")
        download_file(python_url, installer_path)
        subprocess.run([installer_path, "/quiet", f"TargetDir={target_dir}", "InstallAllUsers=0", "PrependPath=1"],
                       check=True)
        os.remove(installer_path)
        python_exe = os.path.join(target_dir, "python.exe")
    else:
        python_url = f"https://www.python.org/ftp/python/{version}/Python-{version}.tgz"
        archive_path = os.path.join(target_dir, "python.tgz")
        download_file(python_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target_dir) # nosec: S202
        os.remove(archive_path)
        python_extracted = os.path.join(target_dir, f"Python-{version}")
        subprocess.run(["./configure", "--prefix=" + target_dir], cwd=python_extracted)
        subprocess.run(["make", "install"], cwd=python_extracted)
        python_exe = os.path.join(target_dir, "bin", "python3")
    return python_exe


def pip_exists(python_exe):
    try:
        subprocess.run([python_exe, "-m", "pip", "--version"], stdout=subprocess.DEVNULL)
        print("pip already installed.")
        return True
    except subprocess.CalledProcessError:
        return False


def install_pip(python_exe):
    if pip_exists(python_exe):
        return
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_path = os.path.join(os.path.dirname(python_exe), "get-pip.py")
    download_file(get_pip_url, get_pip_path)
    subprocess.run([python_exe, get_pip_path])
    os.remove(get_pip_path)


def install_current_package(python_exe):
    subprocess.run([python_exe, "-m", "pip", "install", "-e", ".[isaa]", "--no-warn-script-location"])

def install_extra_win_gpu_12_6_package(python_exe):
    subprocess.run([python_exe, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu126"])
    subprocess.run([python_exe, "-m", "pip", "install", "starlette",
    "litellm",
    "pebble",
    "transformers",
    "networkx",
    "numpy",
    "tiktoken",
    "chromadb",
    "openai",
    "pyaudio",
    "whisper",
    "gtts",
    "pydub",
    "websockets",
    "keyboard",
    "pyperclip",
    "pygments",
    "beautifulsoup4",
    "duckduckgo_search",
    "langchain",
    "langchain_community",
    "langchain_core",
    "Pebble",
    "Requests",
    "torch",
    "tqdm",
    "utils",
    "tokenizers",
    "sentence-transformers",
    "browser-use>=0.1.40",
    "python-git>=2018.2.1",
    "langchain-experimental>=0.3.4",
    "rapidfuzz>=3.12.2",
    "astor>=0.8.1",
    "taichi>=1.7.3",
    "nest-asyncio>=1.6.0",
    "schedule>=1.2.2",
    "qdrant-client[fastembed]",
    "python-levenshtein>=0.27.1",
    "langchain-google-genai>=2.1.2"])
    subprocess.run([python_exe,
                    "-m",
                    "pip",
                    "install",
                    "websockets",
                    "schedule",
                    "mailjet_rest",
                    "mockito"])


def main():
    target_dir = os.path.abspath("python_env")
    os.makedirs(target_dir, exist_ok=True)

    python_exe = install_python(target_dir)
    install_pip(python_exe)
    # time.sleep(5)
    install_current_package(python_exe)
    install_extra_win_gpu_12_6_package(python_exe)
    print(f"✅ Python installed in {target_dir}")
    print(f"✅ Run your Python with: {python_exe}")


if __name__ == "__main__":
    main()
