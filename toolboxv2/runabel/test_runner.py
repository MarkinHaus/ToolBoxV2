"""Console script for toolboxv2. Isaa CMD Tool"""
from toolboxv2.utils.toolbox import ApiOb

NAME = "test"


def run(app, args):
    vt = app.inplace_load("VirtualizationTool", "toolboxv2.mods_dev.")
    vt.create_instance("test-user-root-isaa-instance-v", 'isaa')

    vt.list_instances()

    vt.set_ac_instances("test-user-root-isaa-instance-v")
    data = ApiOb()
    data.data = {'name':'isaa-chat-web','text':'was ergibt 15 + 5'}
    print("#" * 10)
    print(app.run_any("isaa", 'api_run', [data,]))
    print("#" * 20)
    data.data = {'name':'isaa-chat-web','text':' + 5'}
    print("#"*10)
    print(app.run_any("isaa", 'api_run', [data,]))
    print("#" * 20)
    vt.get_instance("test-user-root-isaa-instance-v").name = 'test-user-i'

    vt.list_instances()

    vt.set_ac_instances("test-user-root-isaa-instance-v")

    data.data = {'name':'isaa-chat-web','text':' - 5'}
    print("#" * 10)
    print(app.run_any("isaa", 'api_run', [data,]))
    print("#" * 20)
