from toolboxv2 import get_app

Name = 'CloudM.settings'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
test_only = export(mod_name=Name, test_only=True, state=False)
version = '0.0.1'

# log in with jwt get instance id

