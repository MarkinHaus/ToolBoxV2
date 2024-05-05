import logging
import os
import tempfile
import time

try:
    import docker
    from docker.models.containers import Container

    DOCKER = True
except ModuleNotFoundError:
    DOCKER = False

from toolboxv2 import MainTool, FileHandler, App, Style, Spinner, get_app


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.2"
        self.name = "dockerEnv"
        self.logger: logging.Logger or None = app.logger if app else None
        self.color = "WHITE"
        self.keys = {}
        self.tools = {
            "all": [["Version", "Shows current Version"], ["build_custom_images", "Shows current Version"]],
            "name": "dockerEnv",
            "Version": self.show_version,
            "build_custom_images": self.build_custom_images,
        }

        self.client = None
        if DOCKER:
            self.client = docker.from_env()
        self.container_dict = {}
        self.streaming_function = lambda x: print(str(x, 'utf-8'))
        self.command_prefix, self.command_post_fix = '/bin/sh -c "', '"'

        FileHandler.__init__(self, "dockerEnv.config", app.id if app else __name__, keys=self.keys, defaults={})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logger, color=self.color, on_exit=self.on_exit)

    def on_start(self):
        self.logger.info(f"Starting dockerEnv")
        self.load_file_handler()

    def on_exit(self):
        self.logger.info(f"Closing dockerEnv")
        self.save_file_handler()

    def show_version(self):
        self.print("Version: ", self.version)
        return self.version

    def get_all_containers(self):
        containers = self.client.containers.list(all=True)
        for container in containers:
            self.print(f"Container: {container.name}")

    def build_custom_images(self, img_name: str, file_data: str or None = None, path: str or None = None):

        if path is not None:
            self.print("Using existing file")
            temp_path = path
        elif file_data is not None:
            self.print("Cratering temporary file")
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
                if self.app.system_flag != "Windows":
                    temp_path = "/var" + temp_path
        else:
            raise ValueError("Optional pars file_data or path not both")

        with Spinner(message=f"Building Docker image name: {img_name}", symbols="c"):
            os.system(f"docker build --no-cache -f {temp_path} -t {img_name} .")

        if file_data is not None and os.path.exists(temp_path):
            os.remove(temp_path)

        return img_name

    def start_container(self, name):
        container = self.get_container(name)
        self.print(f"Starting container {container.name}:{container.status}")
        if container.status == "paused":
            container.unpause()
        elif container.status != "running":
            container.start()
        return container

    def reload_container(self, name):
        container = self.get_container(name)
        container.reload()
        return container

    def delete_container(self, name):
        container = self.get_container(name)
        container.stop()
        container.kill()

    def stop_container(self, name):
        container = self.get_container(name)
        container.stop()

    def pause_container(self, name):
        container = self.get_container(name)
        container.pause()

    def create_container(self, image, name, ports=None, entrypoint=None):

        if entrypoint is None:
            entrypoint = ["tail", "-f", "/dev/null"]

        container = None

        try:
            container = self.start_container(name)
        except Exception:
            pass

        if container is None:
            container = self.client.containers.run(image, name=name, detach=True, entrypoint=entrypoint, ports=ports)

            self.print(f"Running container : Status {container.status} E {entrypoint}")

        return container.id

    def init_with_git_repo(self, container, git_repo_url):
        command = f"git clone {git_repo_url}"
        if isinstance(container, str):
            container = self.get_container(container)
        exit_code, output = container.exec_run(command, stdout=True, stderr=True)
        if exit_code == 0:
            self.commit_container(container, f"Initialized with Git Repo from {git_repo_url}")
            return
        self.logger.warning(
            f"Git repository was not initialized properly starts code: {exit_code} informations: {output}")

    def run_command_in_container(self, container, command, stream=False):
        if isinstance(container, str):
            container = self.get_container(container)

        if container.status != "running":
            self.logger.warning("Container not running try auto restart")
            container = self.start_container(container.id)

            time.sleep(1.4)

        if container.status == "running":

            command = self.command_prefix + command + self.command_post_fix
            self.print(command)
            output = ""
            if stream:
                exit_code, gen = container.exec_run(command, stdout=True, stderr=True, stream=True)

                for output in gen:
                    self.streaming_function(output)
            else:
                exit_code, output = container.exec_run(command, stdout=True, stderr=True)

            self.print(f"Command : {exit_code} {output}")
            if exit_code == 0:
                self.commit_container(container, f"Ran command: {command}")
            return output

        self.logger.error(f"Unable to start the Container {container.status}")

    def get_container(self, container_id) -> Container:
        container = self.client.containers.get(container_id)
        return container

    def commit_container(self, container, message):

        self.print(container.diff())

        # commit_id = container.commit(message=message)
        # return commit_id

    def save_changes(self, container, message):

        # get_archive
        # put_archive

        self.print(container.get_archive())

        commit_id = container.commit(message=message)
        return commit_id

    def session_cli_user(self, container_id, stream=True, input_fuction=None):

        if input_fuction is None:
            input_fuction = input

        container = self.get_container(container_id)

        if container.status != "running":
            self.start_container(container_id)

        running = True

        while running:

            user_input = input_fuction()

            if user_input == "exit":
                running = False
                break

            try:
                self.run_command_in_container(container, user_input, stream=stream)
            except Exception as e:
                self.print(e)
                running = False

            self.print(container.diff())

        self.stop_container(container_id)


if __name__ == "__main__":
    docker = get_app(from_="test_docker", name="debug").save_load("dockerEnv")
