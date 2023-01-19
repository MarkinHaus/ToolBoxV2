from typing import Union
from pydantic import BaseModel
from datetime import datetime
import uuid

from toolboxv2 import MainTool, FileHandler, App, Style


class Tools(MainTool, FileHandler):

    def __init__(self, app=None):
        self.version = "0.0.1"
        self.name = "daytree"
        self.logs = app.logs_ if app else None
        self.color = "BEIGE2"
        self.keys = {"Config": "config~~~:",
                     "Bucket": "bucket~~~:"}
        self.config = {}
        self.tools = {
            "all": [["Version", "Shows current Version"],
                    ["designer_input", "Day Tree designer input Stream"],
                    ["save_task_to_bucket", "Day Tree designer jo"],
                    ["get_bucket_today", "Day Tree designer jo"],
                    ["get_bucket_week", "Day Tree designer jo"],
                    ["save_task_day", "Day Tree designer jo"],
                    ["save_task_week", "Day Tree designer jo"],
                    ["due_kwd", "Day Tree designer jo"],
                    ],
            "name": "daytree",
            "Version": self.show_version,
            "designer_input": self.designer_input,
            "save_task_to_bucket": self.save_task_to_bucket,
            "get_bucket_today": self.get_bucket_today,
            "get_bucket_week": self.get_bucket_week,
            "save_task_day": self.save_task_day,
            "save_task_week": self.save_task_week,
            "due_kwd": self.due_date_to_kwd,
        }
        FileHandler.__init__(self, "daytree.config", app.id if app else __name__, self.keys, {
            "Config": {"Modi": ["Ich mus um eine bestimmte Uhrzeit an einem bestimmten Ort mit oder ohne weitere "
                                "Personen Besuchen, es handelt sich um einen Termin.",
                                "Ich möchte mich an eine Sache oder Tätigkeit erinnern, es handelt sich um eine "
                                "Erinnerung.",
                                "Ich muss eine Bestimmte aufgebe Erledigen, es handelt sich um eine Aufgabe."],

                       'vg_ob': {'time': ['time', 'uhr', 'zeit'],
                                 'due_date': ['due_date', 'datum', 'date'],
                                 'day': ['tag', 'day'],
                                 'week': ['week', 'kw'],
                                 'priority': ['priority', 'P#', '!'],
                                 'cal': ['cal'], }
                       },
            "Bucket": {}})
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools,
                          name=self.name, logs=self.logs, color=self.color, on_exit=self.on_exit)

    def show_version(self):
        self.print("Version: ", self.version)

    def on_start(self):
        self.load_file_handler()
        self.config = self.get_file_handler(self.keys["Config"])

    def on_exit(self):
        self.add_to_save_file_handler(self.keys["Config"], str(self.config))
        self.save_file_handler()
        self.file_handler_storage.close()

    def designer_input(self, command, app: App):
        if "isaa" not in list(app.MOD_LIST.keys()):
            return "Server has no isaa module"
        if len(command) > 2:
            return {"error": f"Command-invalid-length {len(command)=} | 2 {command}"}

        uid, err = self.get_uid(command, app)

        if err:
            return uid

        data = command[0].data
        self.print(data["input"])

        att_list = []
        att_test = {'v': 'test', 't': 'test'}
        att_list.append(att_test)
        return att_list

    def save_task_to_bucket(self, command, app: App):

        uid, err = self.get_uid(command, app)
        if err:
            return uid

        self._load_save_db(app, f"bucket::{uid}", [command[0].data["task"]])

        return "Don"

    def _load_save_db(self, app: App, db_key, data):
        bucket = app.run_any('db', 'get', [f"dayTree::{db_key}"])
        if bucket == "":
            bucket = []
        else:
            bucket = eval(bucket)

        for elm in data:
            bucket.append(elm)

        app.run_any('db', 'set', ["", f"dayTree::{db_key}", str(bucket)])
        return bucket

    def _dump_bucket(self, app: App, uid):

        bucket = app.run_any('db', 'get', [f"dayTree::bucket::{uid}"])  # 1 bf bl
        if bucket == "":
            bucket = []
        else:
            try:
                bucket = eval(bucket)
            except TypeError:
                return "bucket-error-eval failed"
            app.run_any('db', 'del', ["", f"dayTree::bucket::{uid}"])

        self.print("bucket - len : ", len(bucket))

        tx, wx = self._sort_tx_wx(bucket)
        return self._append_tx_wx(app, uid, tx, wx)

    def _sort_tx_wx(self, bucket):
        wx, tx = [], []

        for task in bucket:
            wx_task = self._wx_format_task(task)
            cal = self._calculate_cal(wx_task, [0, 0])
            if cal > 0:
                wx.append(wx_task)
            else:
                tx.append(task)

        return tx, wx

    def _wx_format_task(self, task):
        # -> {name: "dings", }

        # Lambda-Funktion zum Hinzufügen von Eigenschaften zu einem Task-Objekt
        add_properties = lambda task_, vg_ob: {
            'id': str(uuid.uuid4()).replace('-', ''),  # Zufällige UID generieren

            # Schleife über die Eigenschaften in vg_ob
            **{prop: next(filter(lambda x: x['t'] in vg_ob[prop], task_['att']), {'v': 0})['v']
               for prop in vg_ob.keys()},

            **task_  # Restliche Eigenschaften aus task übernehmen
        }

        return add_properties(task, self.config['vg_ob'])

    def _calculate_cal(self, item, r):

        if not item:
            return -1
        try:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            priority = item['priority'] if 'priority' in item.keys() else 0
            kw = int(
                datetime.strptime(str(item['due_date']).replace(".", "-").replace(" ", "-").replace(":", "-"),
                                  '%Y-%m-%d').strftime('%W')) if 'due_date' in item.keys() and str(
                item['due_date']) != "0" else (
                item['week'] if 'week' in item.keys() else 0)
            day = int(days.index((
                datetime.strptime(str(item['due_date']).replace(".", "-").replace(" ", "-").replace(":", "-")
                                  , '%Y-%m-%d').strftime('%A')))) if 'due_date' in item.keys() and str(
                item['due_date']) != "0" else (
                item['day'] if 'day' in item.keys() else -1)

            now = datetime.now()
            # Kalenderwoche und Tag aus dem Datetime-Objekt extrahieren
            day_rel = (day - days.index(now.strftime('%A'))) + r[0]  # day_now
            kw_rel = (kw - int(now.strftime('%W'))) + r[1]  # kw_now
            if kw == 0:
                kw_rel = 0
            if day == -1:
                kw_rel = 0
            self.print(
                f"{priority = } {day_rel = }-{day}-{days.index(now.strftime('%A'))} | {kw_rel = }-{kw}-{int(now.strftime('%W'))} | rel-task-now")
            c = 1 + int(priority) + int(day_rel) * 2.5 + (int(kw_rel) * 10)
            self.print(f"{priority} + {day_rel * 2.5} + {kw_rel * 10} +1 -> {c}")
            return c
        except ValueError:
            return -1

    def _sort_wx(self, wx, r):
        if r is None:
            r = [0, 0]
        todo_list_formatted = [
            {
                'name': item['name'] if 'name' in item.keys() else "#None-TASK#",
                'index': i,
                'id': item['id'] if 'id' in item.keys() else 0,
                'priority': item['priority'] if 'priority' in item.keys() else 0,
                'cal': self._calculate_cal(item, r),
            }
            for i, item in enumerate(wx)
        ]
        key = lambda x: (x['cal'] < 0, x['cal'], (x[
                                                      'time'] / 100 if 'time' in x.keys() else 0))  # lambda x: (x['cal'], (x['time'] / 100 if 'time' in x.keys() else 0))

        sorted_list = sorted(todo_list_formatted,
                             key=key)
        return sorted_list

    def due_date_to_kwd(self, command, app: App):

        data = command[0].data
        uid, err = self.get_uid(command, app)

        if err:
            return uid

        due_date = data["due_date"]
        # Fälligkeitsdatum als Datetime-Objekt umwandeln
        due_date_dt = datetime.strptime(str(due_date), '%Y-%m-%d')

        # Kalenderwoche und Tag aus dem Datetime-Objekt extrahieren
        week = due_date_dt.strftime('%W')
        day = due_date_dt.strftime('%A')
        return week, day

    def _append_tx_wx(self, app, uid, tx, wx):
        wx = self._load_save_db(app, f"wx::{uid}", wx)
        tx = self._load_save_db(app, f"tx::{uid}", tx)
        return wx, tx

    def _get_day_x(self, wx, tx, x):
        ts = []

        for _x in wx:
            _x = self._wx_format_task(_x)

        wx_now = self._sort_wx(wx, [x[0], x[1]])
        if x[0] != 0:
            if len(tx) > x[2] - 1:
                ts.append(tx[:x[2]])
                tx = tx[:x[2]]
            else:
                for t in tx:
                    ts.append(t)
                tx = []
        print(f"{wx_now=} {wx=}")
        if len(wx_now) > x[2] - 1:
            # Get the IDs of the first x[2] elements in wx_now
            wx_now_x_ids = [item["index"] for item in wx_now[:x[2]]]

            # Lambda function to filter the task objects in wx by ID
            filter_tasks = lambda ids, tasks: list(filter(lambda task: task['index'] in ids, tasks))

            # Filter the tasks in wx based on the IDs in wx_now_x_ids
            wx_x = filter_tasks(wx_now_x_ids, wx)

            # Set the "cal" attribute of each task in wx_x to the corresponding task in wx_now
            for task_x, task_now in zip(wx_x, wx_now):
                del task_x['cal']
                task_x['att'].append({'t': 'cal', 'v': task_now['cal']})

            # Append the filtered tasks to ts
            print(wx_x)
            ts.append(wx_x)

            # Remove the tasks in wx_x from wx
            for task in wx_x:
                wx.remove(task)

        else:
            for w in wx:
                ts.append(w)
            wx = []

        return ts, wx, tx

    def get_bucket_today(self, command, app: App):
        uid, err = self.get_uid(command, app)
        if err:
            return uid

        day = app.run_any('db', 'get', [f"dayTree::bucket::{uid}"])
        if day == "":
            day = []
        else:
            day = eval(day)

        if len(day) == 0:
            wx, tx = self._dump_bucket(app, uid)
            day, _, _ = self._get_day_x(wx, tx, [0, 0, 10])

        return day

    def get_bucket_week(self, command, app: App):
        uid, err = self.get_uid(command, app)
        if err:
            return uid

        wx, tx = self._dump_bucket(app, uid)
        week = []
        print(f"{tx=}\n{wx=}")
        for i in range(1, 8):
            week.append([])
            if len(wx) != 0 or len(tx) != 0:
                day, wx, tx = self._get_day_x(wx, tx, [i - 1, 0, 10])
                print(f"{tx=}\n{wx=}\n{day=}")
                for t in day:
                    week[i - 1].append(t)

        return week

    def get_day(self, command, app: App):

        uid, err = self.get_uid(command, app)
        if err:
            return uid

        day_date = command[0].data["date"]

        wx, tx = self._dump_bucket(app, uid)
        day, wx, tx = self._get_day_x(wx, tx, [int(day_date), 0, 10])

    def get_uid(self, command, app: App):
        if "cloudm" not in list(app.MOD_LIST.keys()):
            return "Server has no cloudM module"

        if "db" not in list(app.MOD_LIST.keys()):
            return "Server has no database module"

        res = app.run_any('cloudm', "validate_jwt", command)

        if isinstance(res, str):
            return res, True

        if not isinstance(res, dict):
            return res, True

        if "res" in res.keys():
            res = res["res"]

        if "uid" not in res.keys():
            return res, True

        return res["uid"], False

    def save_task_day(self, command, app: App):

        data = command[0].data
        uid, err = self.get_uid(command, app)

        if err:
            return uid

        day = data["task"]

        if len(day) == 0:
            wx, tx = self._dump_bucket(app, uid)
            _, wx, tx = self._get_day_x(wx, tx, [0, 0, 10])
            day, wx, tx = self._get_day_x(wx, tx, [0, 0, 10])
            self.print(app.run_any('db', 'set', ["", f"dayTree::tx::{uid}", str(tx)]))
            self.print(app.run_any('db', 'set', ["", f"dayTree::wx::{uid}", str(wx)]))
            self.print(app.run_any('db', 'set', ["", f"dayTree::day::{uid}", str(day)]))
        elif len(day) == 10:
            return day
        else:
            wx, tx = [], []
            _tx, _wx = self._sort_tx_wx(day)
            for t in _tx:
                tx.append(t)
            for w in _wx:
                wx.append(w)

            self.print(app.run_any('db', 'set', ["", f"dayTree::tx::{uid}", str(tx)]))
            self.print(app.run_any('db', 'set', ["", f"dayTree::wx::{uid}", str(wx)]))

        self.print(app.run_any('db', 'set', ["", f"dayTree::day::{uid}", str(day)]))
        return day

    def save_task_week(self, command, app: App):

        data = command[0].data
        uid, err = self.get_uid(command, app)

        if err:
            return uid

        week = data["week"]
        self.print("Lazy save_task_week")
        # tx = []
        # for i in range(0, 7):
        #     tx = self.r_twx(app, uid)

        tx, wx = [], []
        for i, day in enumerate(week):
            self.print(f"sorting:{i} - len {len(day)}")
            _tx, _wx = self._sort_tx_wx(day)
            for t in _tx:
                tx.append(t)
            for w in _wx:
                wx.append(w)

        self.print(app.run_any('db', 'set', ["", f"dayTree::tx::{uid}", str(tx)]))
        self.print(app.run_any('db', 'set', ["", f"dayTree::wx::{uid}", str(wx)]))
        self.print(app.run_any('db', 'set', ["", f"dayTree::week::{uid}", str(week)]))

        return "week"

    def r_twx(self, app, uid):
        tx = self._get_twx("t", app, uid)
        if len(tx) >= 10:
            tx = tx[:10]
            return tx
        return []

    def _get_twx(self, c, app, uid):
        cx = app.run_any('db', 'get', [f"dayTree::{c}x::{uid}"])
        self.print(f"{c}x={cx}")
        if cx == "":
            cx = []
        else:
            cx = eval(cx)
        return cx
