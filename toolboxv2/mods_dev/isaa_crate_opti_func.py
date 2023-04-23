from toolboxv2 import MainTool, App


def list_s_str_f(s):
    if isinstance(s, str):
        if s[0] != '/':
            s = '/' + s
        return s
    if len(s) == 0:
        return ""
    if len(s) == 1:
        return s[0]
    return s[1]


class Tools(MainTool):
    def __init__(self, app=None):
        self.version = "0.1"
        self.name = "isaa_ide"
        self.logs = app.logger if app else None
        self.color = "YELLOW"
        self.tools = {
            "all": [
                ["create", ":Beschreibung:"],
            ],
            "create": ["create(path)", "Create a file or directory at the specified path."],
        }

        self.scope = "./isaa-directory"

        # Steps:
        #   1. -> Get data
        #   2. -> get relevant information's
        #   3. -> Think about optimisations staps
        #       3.1 -> Fehler fangen
        #       3.2 -> Code schneller machen
        #       3.3 -> Besser lesbar
        #       3.4 -> Effizienter
        #       3.5 -> Fehler behebung
        #   4. -> Optimise code
        #   5. -> save code
        #   6. -> repeat

        MainTool.__init__(self, load=None, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=lambda: "")

    def get_data(self, path, app: App):
        app.new_ac_mod("isaa_ide")
        path = list_s_str_f(path)
        return app.AC_MOD.read_file(path)

    def get_relevant_information(self, ):

        return

    def create(self, path, app: App):
        p = list_s_str_f(path)
        path = self.scope + p
        if p[0] != '/':
            path = self.scope + '/' + p
        """
        Create a file or directory at the specified path.
        """
