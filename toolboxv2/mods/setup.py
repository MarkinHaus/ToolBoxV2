"""
Main task of this Mod :
0) from 0 device to base TB (user)
1) from 0 device to devEnv (dev)
2) from devEnv0 to core0 HMR 0 downtime updates (admin)
steps :
    0) :
     * shell - setup python and env
     * shell    # on linux and mac tet install python3-venv
     * shell    # using a mini forward script as base installation calld tb taht runs the toolboxv2 in the venv
     * shell - install ex dependency (node, docker, ...)
     * shell    # using winget or apt or co ...
     * shell - install ToolBoxV2 from git or pip installation dependent
     * shell - init auth connection / login to core0
     * shell    # in mini forward script save and crate session + pars to venv
      #0 installation from mod_sto
        - core from pip or git 0 exras (isaa qn dyt ttt diff alert cyr bkt shm auto)
      #0 test local
    1) :
     - installation devModStorage from core0
     - setup files (dev mods)
     - test local Core
     -> install (MOD-NAME)
        install from .yaml file
         specs: name
                version
                dependencies
                extras
   2) :
     -# development (?ISAA?)
        -~ uploade to devModStorage
       -> test local
       -> test remote
     -> move to mod_sto
     -> install on to remote
"""
from toolboxv2 import get_app

Name = 'setup'
export = get_app("setup.Export").tb
default_export = export(mod_name=Name)
version = '0.0.3'
spec = ''


@export(mod_name=Name, name='Version', version=version)
def get_version():
    return version


"""
Architecture :: State transform via Running Scenarios

:: States ::
 '' dev0
 '' P0/S0
 '' PN/SN
:: Phases ::
-> setup
-> build
-> deploy

:: Scenarios ::
[Ich bin] 'und' [ich,du werde]
 -> meine aktionen um den neuen zustand zu erreichen

 dev0 '' P0/S0
  -> test
  -> build
  -> test
  -> deploy

 P0/S0 '' PN/SN
  -> deploy

"""
