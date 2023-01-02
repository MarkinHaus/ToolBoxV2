# ToolBoxV2


[![image](https://img.shields.io/pypi/v/ToolBoxV2.svg)](https://pypi.python.org/pypi/ToolBoxV2)
[![image](https://img.shields.io/conda/vn/conda-forge/ToolBoxV2.svg)](https://anaconda.org/conda-forge/ToolBoxV2)

[![image](https://pyup.io/repos/github/MarkinHaus/ToolBoxV2/shield.svg)](https://pyup.io/repos/github/MarkinHaus/ToolBoxV2)
[![image](https://img.shields.io/badge/Donate-Buy%20me%20a%20coffee-yellowgreen.svg)](https://pyup.io/repos/github/MarkinHaus/ToolBoxV2)


** Command line interface for interactions with the ToolBox Network.  **

-   License: Apache Software License 2.0


Installation
------------

    pip install ToolBoxV2


-  Github
```
git clone git://github.com/MarkinHaus/ToolBoxV2
```

Usage
-----

    $ toolboxv2 -h
    usage: toolboxv2 [-h] [-init INIT] [-f INIT_FILE] [-update] [--update-mod UPDATE_MOD] [--delete-ToolBoxV2 {allcli,dev,api,config,data,src,all} {allcli,dev,api,config,data,src,all}] [--delete-mod DELETE_MOD] [-v] [-mvn name]
                 [-n name] [-m {cli,dev,api,app}] [-p port] [-w host] [-l]

    Welcome to the ToolBox cli

    options:
      -h, --help            show this help message and exit
      -init INIT            ToolBoxV2 init (name) -> default : -n name = main
      -f INIT_FILE, --init-file INIT_FILE
                            optional init flag init from config file or url
      -update               update ToolBox
      --update-mod UPDATE_MOD
                            update ToolBox mod
      --delete-ToolBoxV2 {allcli,dev,api,config,data,src,all} {allcli,dev,api,config,data,src,all}
                            delete ToolBox or mod | ToolBoxV2 --delete-ToolBoxV2
      --delete-mod DELETE_MOD
                            delete ToolBoxV2 mod | ToolBox --delete-mod (mod-name)
      -v, --get-version     get version of ToolBox | ToolBoxV2 -v -n (mod-name)
      -mvn name, --mod-version-name name
                            Name of mod
      -n name, --name name  Name of ToolBox
      -m {cli,dev,api,app}, --modi {cli,dev,api,app}
                            Start ToolBox in different modes
      -p port, --port port  Specify a port for dev | api
      -w host, --host host  Specify a host for dev | api
      -l, --load-all-mod-in-files


Setup
----------
Set up main
~~~~~~~~~~~~~~~~~~~
ToolBoxV2 -init main -f init.config
~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~DESKTOP-GIT@>Exit
    ~DESKTOP-GIT@>y

- Set up mod settings config / data
~~~~~~
ToolBoxV2 -init {mod name} -n {name} -f {loc of init file}
~~~~~~~~~~~~~~~~~~~

First Start
----------
~~~~~~~~~~~~~~~~~~~
    $ ToolBoxV2 || ToolBoxV2 -n {name}
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
    ~DESKTOP-GIT@>load-mod cloudM
    ~DESKTOP-GIT:CLOUDM@>create-account
~~~~~~~~~~~~~~~~~~~~~~~~~~
- enabling all mods in mods folder
~~~~~~
    $ ToolBoxV2 -l || ToolBoxV2 -n {name} -l
~~~~~~~~~~~~~~~~~~~~~~~~~~
in TB:
~~~~~~
cloudM create-account
~~~~~~~~~~~~~~~~~~~~~~~~~~
- or
~~~~~~
~DESKTOP-GIT@>cloudM
~DESKTOP-GIT:CLOUDM@>create-account
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Log in
~~~~~~
~DESKTOP-GIT:CLOUDM@>login {username} {password}
~~~~~~~~~~~~~~~~~~~~~~~~~~
- or ( in v 0.0.2)
~~~~~~
~DESKTOP-GIT:CLOUDM@>login {token}
~~~~~~~~~~~~~~~~~~~~~~~~~~

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.
