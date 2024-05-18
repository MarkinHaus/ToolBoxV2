from toolboxv2 import MainTool
from toolboxv2.utils.extras import BaseWidget

Name = "WidgetsProvider.BoardWidget"
version = "0.0.1"

template = """

<title>Board Management</title>
<style>
/* CSS für die Tabelle */
table {
  width: 100%;
  border-collapse: collapse;
}

table, th, td {
  border: 1px solid black;
  padding: 8px;
}

th {
  background-color: #f2f2f2;
}

button {
  padding: 5px 10px;
  background-color: #4CAF50;
  color: white;
  border: none;
  cursor: pointer;
}

button:hover {
  background-color: #45a049;
}
</style>
</head>
<body>

<h1 id="boardTitle">Board Management</h1>
<br/>
<h2 id="inEdit"> In Edit : mainboard </h2>
<table id="boardTable">
  <thead>
    <tr>
      <th>Name</th>
      <th>Open/Close</th>
      <th>Set Edit</th>
      <th>Delete</th>
    </tr>
  </thead>
  <tbody>
    <!-- Hier werden die Zeilen der Tabelle dynamisch eingefügt -->
  </tbody>
</table>

<script>
// JavaScript für das Erstellen der Tabelle aus einer Liste von Board-Namen
if(window.history.state.TB) {
  // Beispiel-Liste von Board-Namen
  var boardNames = $boardNames;
  var tableBody = document.querySelector('#boardTable tbody');

  // Funktion zum Hinzufügen von Zeilen zur Tabelle
  function addRow(name) {
    var row = tableBody.insertRow();
    row.innerHTML = `
      <td>${name}</td>
      <td><input type="checkbox"></td>
      <td><input type="checkbox"></td>
      <td><button onclick="confirmDelete('${name}')">Delete</button></td>
    `;
  }

  // Durchlaufe die Liste von Board-Namen und füge sie zur Tabelle hinzu
  boardNames.forEach(function(name) {
    addRow(name);
  });

  // Funktion zur Bestätigung der Löschung eines Boards
  window.confirmDelete = function(name) {
    if (confirm(`Are you sure you want to delete "${name}" board?`)) {
      // Hier kann die Lösch-Logik implementiert werden
      console.log(`${name} board deleted`);
    }
  };
};
</script>
"""


class BoardWidget(MainTool, BaseWidget):

    def __init__(self, app=None):
        self.name = Name
        self.color = "WITHE"
        self.tools = {'name': Name}
        self.version = version
        MainTool.__init__(self,
                          load=self.on_start,
                          v=self.version,
                          name=self.name,
                          color=self.color,
                          on_exit=self.on_exit)

        BaseWidget.__init__(self, name=self.name)
        self.register(self.app, self.get_widget, self.version)

    def main(self, request):
        w_id = self.get_s_id(request)
        if w_id.is_error():
            return w_id
        self.asset_loder(self.app, self.name, w_id.get(), template=template, boardNames=["pBoard 1",
                                                                                         "pBoard 2",
                                                                                         "pBoard 3"])

    def on_start(self):
        self.register2reload(
            self.main
        )

    def on_exit(self):
        pass

    async def get_widget(self, request):
        w_id = self.get_s_id(request)
        if w_id.is_error():
            return w_id
        return self.load_widget(self.app, request, self.name, w_id.get())


