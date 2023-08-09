import logging
import os
from platform import system

import requests

from toolboxv2 import MainTool, FileHandler, App


class Tools(MainTool, FileHandler):  # FileHandler

  def __init__(self, app=None):
    self.version = "0.0.2"
    self.name = "api_manager"
    self.logger: logging.Logger = app.logger if app else None
    self.color = "WHITE"
    self.keys = {"Apis": "api~config"}
    self.api_pid = None
    self.api_config = {}
    self.tools = {
      "all": [
        ["Version", "Shows current Version"],
        ["edit-api", "Set default API for name host port "],
        ["start-api", ""],
        ["stop-api", ""],
        ["restart-api", ""],
        ["info", ""],
      ],
      "name":
      "api_manager",
      "Version":
      self.show_version,
      "edit-api":
      self.conf_api,
      "start-api":
      self.start_api,
      "stop-api":
      self.stop_api,
      "info":
      self.info,
      "restart-api":
      self.restart_api,
    }
    FileHandler.__init__(
      self, "api-m.data", app.id if app else __name__, self.keys, {
        "Apis": {
          'main': {
            "Name": 'main',
            "version": self.version,
            "port": 5000,
            "host": '127.0.0.1'
          }
        }
      })
    MainTool.__init__(self,
                      load=self.on_start,
                      v=self.version,
                      tool=self.tools,
                      name=self.name,
                      logs=self.logger,
                      color=self.color,
                      on_exit=self.on_exit)

  def show_version(self):
    self.print("Version: ", self.version)
    return self.version

  def info(self):
    for api in list(self.api_config.keys()):
      self.print(f"Name: {api}")
      self.print(self.api_config[api])
    return self.api_config

  def conf_api(self, command):
    if len(command) <= 4:
      return "invalid command length [api:name host port]"
    host = command[2]
    if host == "lh":
      host = "127.0.0.1"
    if host == "0":
      host = "0.0.0.0"
    port = command[3]
    if port == "0":
      port = "8000"
    self.api_config[command[1]] = {
      "Name": command[1],
      "version": self.version,
      "port": port,
      "host": host
    }

    self.print(self.api_config[command[1]])

  def start_api(self, command, app: App):
    if len(command) == 1:
      command.append('-main-')
    api_name = command[1]

    if api_name not in self.api_config.keys():
      host = "127.0.0.1"
      if 'live' in api_name:
        host = "0.0.0.0"

      self.api_config[api_name] = {
        "Name": api_name,
        "version": self.version,
        "port": 5000,
        "host": host
      }

    api_data = self.api_config[api_name]

    self.print(app.pretty_print(api_data))
    self.print(api_data)
    g = f"uvicorn toolboxv2.api.fast_api_main:app --host {api_data['host']}" \
        f" --port {api_data['port']} --header data:{api_name}.config:{api_name} {'--reload' if 'reload' in app.id else ''}"
    print(g)
    os.system(g)

  def stop_api(self, command):
    if command[1] in list(self.api_config.keys()):
      command.append(self.api_config[command[1]]['host'])
      command.append(self.api_config[command[1]]['port'])
    else:
      if len(command) == 2:
        command += ["127.0.0.1", 5000]

    if not os.path.exists(f"./.data/api_pid_{command[1]}"):
      self.logger.warning("no api_pid file found ")
      return
    with open(f"./.data/api_pid_{command[1]}", "r") as f:
      api_pid = f.read()
      requests.get(f"http://{command[2]}:{command[3]}/api/exit/{api_pid}")
      if system() == "Windows":
        os.system(f"taskkill /pid {api_pid} /F")
      else:
        os.system(f"kill -9 {api_pid}")
      #os.remove(f"app/api_pid_{command[1]}")

  def restart_api(self, command, app: App):
    self.stop_api(command.copy())
    self.start_api(command, app)

  def on_start(self):
    self.load_file_handler()
    self.api_config = self.get_file_handler(self.keys["Apis"])

  def on_exit(self):
    self.add_to_save_file_handler(self.keys["Apis"], str(self.api_config))
    self.save_file_handler()
