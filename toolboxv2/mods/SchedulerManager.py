import json
import os
import pickle
from typing import Callable, Optional

try:
    import dill
except ImportError:
    dill = pickle

import schedule
import threading
import time
from datetime import datetime, timedelta
from toolboxv2 import get_app, App, Result, MainTool, FileHandler

Name = 'SchedulerManager'
export = get_app(Name).tb
version = '0.0.2'

safety_mode = ['open', 'strict', 'closed'][0]
serializer_default, deserializer_default = [(dill, dill), (dill, dill), (pickle, pickle)] \
    [['open', 'strict', 'closed'].index(safety_mode)]


class SchedulerManagerClass:
    def __init__(self):
        self.jobs = {}
        self.thread = None
        self.running = False

    def _run(self):
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def register_job(self,
                     job_id: str,
                     second: int = -1,
                     func: Optional[Callable or str] = None,
                     job: Optional[schedule.Job] = None,
                     time_passer: Optional[schedule.Job] = None,
                     object_name: Optional[str] = None,
                     receive_job: bool = False,
                     save: bool = False,
                     max_live: bool = False,
                     serializer=serializer_default,
                     args=None, kwargs=None):
        """
            Parameters
            ----------
                job_id : str
                    id for the job for management
                second : int
                    The time interval in seconds between each call of the job.
                func : Callable or str
                    The function to be executed as the job.
                job : schedule.Job
                    An existing job object from the schedule library.
                time_passer : schedule.Job
                    A job without a function, used to specify the time interval.
                object_name : str
                    The name of the object containing in the 'func' var to be executed.
                receive_job : bool
                    A flag indicating whether the job should be received from an object from 'func' var.
                save : bool
                    A flag indicating whether the job should be saved.
                max_live : bool
                    A flag indicating whether the job should have a maximum live time.
                serializer : dill
                    json pickel or dill must have a dumps fuction
                *args, **kwargs : Any serializable and deserializable
                    Additional arguments to be passed to the job function.

            Returns
            -------
           """

        if job is None and func is None:
            raise ValueError("Both job and func are not specified. Please specify either job or func.")
        if job is not None and func is not None:
            raise ValueError("Both job and func are specified. Please specify either job or func.")

        if job is not None:
            func = lambda x: x
            self._save_job(job_id=job_id,
                           job=job,
                           save=save,
                           func=func,
                           args=args,
                           kwargs=kwargs,
                           serializer=serializer)
            return

        try:
            parsed_attr = self._parse_function(func=func,
                                               object_name=object_name)
        except:
            print(f"Error parsing function fro job : {job_id}")
            return

        if receive_job:
            job = parsed_attr
        else:
            func = parsed_attr

        time_passer = self._prepare_time_passer(time_passer=time_passer,
                                                second=second)

        job_func = self._prepare_job_func(func=func,
                                          max_live=max_live,
                                          second=second,
                                          args=args,
                                          kwargs=kwargs,
                                          job_id=job_id)

        job = self._get_final_job(job=job,
                                  func=func,
                                  time_passer=time_passer,
                                  job_func=job_func,
                                  args=args,
                                  kwargs=kwargs)

        self._save_job(job_id=job_id,
                       job=job,
                       save=save,
                       func=func,
                       args=args,
                       kwargs=kwargs,
                       serializer=serializer)

    @staticmethod
    def _parse_function(func: str or Callable, object_name):
        if isinstance(func, str) and func.endswith('.py'):
            with open(func, 'r') as file:
                func_code = file.read()
                exec(func_code)
                func = locals()[object_name]
        elif isinstance(func, str) and func.endswith('.dill') and safety_mode == 'open':
            try:
                with open(func, 'rb') as file:
                    func = dill.load(file)
            except FileNotFoundError:
                raise ValueError(f"Function file {func} not found or dill not installed")
        elif isinstance(func, str):
            local_vars = {'app': get_app(from_=Name + f".pasing.{object_name}")}
            exec(func.strip(), {}, local_vars)
            func = local_vars[object_name]
        elif isinstance(func, Callable):
            pass
        else:
            raise ValueError("Could not parse object scheduler_manager.parse_function")
        return func

    @staticmethod
    def _prepare_time_passer(time_passer, second):
        if time_passer is None and second > 0:
            return schedule.every(second).seconds
        elif time_passer is None and second <= 0:
            raise ValueError(f"second must be greater than 0")
        return time_passer

    def _prepare_job_func(self, func, max_live, second, job_id, *args, **kwargs):
        if max_live:
            end_time = datetime.now() + timedelta(seconds=second)

            def job_func():
                if datetime.now() < end_time:
                    func(*args, **kwargs)
                else:
                    job = self.jobs.get(job_id, {}).get('job')
                    if job is not None:
                        schedule.cancel_job(job)
                    else:
                        print("Error Canceling job")

            return job_func
        return func

    @staticmethod
    def _get_final_job(job, func, time_passer, job_func, args, kwargs):
        if job is None and isinstance(func, Callable):
            job = time_passer.do(job_func, *args, **kwargs)
        elif job is not None:
            pass
        else:
            raise ValueError(f"No Final job found for register")
        return job

    def _save_job(self, job_id, job, save, func, serializer=serializer_default, args=None, **kwargs):
        if job is not None:
            self.jobs[job_id] = {'id': job_id, 'job': job, 'save': save, 'func': serializer.dumps(func), 'args': args,
                                 'kwargs': kwargs}
            print(f"Added Job {job_id} :{' - saved' if save else ''}"
                  f"{' - args ' + str(len(args)) if args else ''}"
                  f"{' - kwargs ' + str(len(kwargs.keys())) if kwargs else ''}")
        else:
            raise ValueError("")

    def cancel_job(self, job_id):
        if job_id not in self.jobs:
            print("Job not found")
            return
        schedule.cancel_job(self.jobs[job_id].get('job'))
        self.jobs[job_id]["cancelled"] = True
        self.jobs[job_id]["save"] = False
        print("Job cancelled")

    def del_job(self, job_id):
        if job_id not in self.jobs:
            print("Job not found")
            return
        if not self.jobs[job_id].get("cancelled", False):
            print("Job not cancelled canceling job")
            self.cancel_job(job_id)
        del self.jobs[job_id]
        print("Job deleted")

    def save_jobs(self, file_path, serializer=serializer_default):
        with open(file_path, 'wb') as file:
            save_jobs = [job for job in self.jobs.values() if job['save']]
            serializer.dump(save_jobs, file)

    def load_jobs(self, file_path, deserializer=deserializer_default):
        with open(file_path, 'rb') as file:
            jobs = deserializer.load(file)
            for job_info in jobs:
                del job_info['job']
                func = deserializer.loads(job_info['func'])
                self.register_job(job_info['id'], func=func, **job_info)


class Tools(MainTool, SchedulerManagerClass):
    version = version

    def __init__(self, app=None):
        self.name = Name
        self.color = "VIOLET2"

        self.keys = {"mode": "db~mode~~:"}
        self.encoding = 'utf-8'
        self.tools = {'name': Name}
        SchedulerManagerClass.__init__(self)
        MainTool.__init__(self,
                          load=self.init_sm,
                          v=self.version,
                          name=self.name,
                          color=self.color,
                          on_exit=self.on_exit)

    @export(
        mod_name=Name,
        name="Version",
        version=version,
    )
    def get_version(self):
        return self.version

    # Exportieren der Scheduler-Instanz für die Nutzung in anderen Modulen
    @export(mod_name=Name, name='init', version=version, initial=True)
    def init_sm(self):
        if os.path.exists(self.app.data_dir + '/jobs.compact'):
            print("SchedulerManager try loading from file")
            self.load_jobs(
                self.app.data_dir + '/jobs.compact'
            )
            print("SchedulerManager Successfully loaded")
        print("STARTING SchedulerManager")
        self.start()

    @export(mod_name=Name, name='clos_manager', version=version, exit_f=True)
    def on_exit(self):
        self.stop()
        self.save_jobs(self.app.data_dir + '/jobs.compact')
        return f"saved {len(self.jobs.keys())} jobs in {self.app.data_dir + '/jobs.compact'}"

    @export(mod_name=Name, name='instance', version=version)
    def get_instance(self):
        return self

    @export(mod_name=Name, name='start', version=version)
    def start_instance(self):
        return self.start()

    @export(mod_name=Name, name='stop', version=version)
    def stop_instance(self):
        return self.stop()

    @export(mod_name=Name, name='cancel', version=version)
    def cancel_instance(self, job_id):
        return self.cancel_job(job_id)

    @export(mod_name=Name, name='dealt', version=version)
    def dealt_instance(self, job_id):
        return self.del_job(job_id)

    @export(mod_name=Name, name='add', version=version)
    def register_instance(self, job_data: dict):
        """
        example dicts :
            -----------
            {
                "job_id": "job0",
                "second": 0,
                "func": None,
                "job": None,
                "time_passer": None,
                "object_name": "tb_job_fuction",
                "receive_job": False,
                "save": False,
                "max_live": True,
                # just lev it out "serializer": serializer_default,
                "args": [],
                "kwargs": {},
            }

            job_id : str
                id for the job for management
            second (optional): int
                The time interval in seconds between each call of the job.
            func (optional): Callable or str
                The function to be executed as the job.
            job (optional):  schedule.Job
                An existing job object from the schedule library.
            time_passer (optional):  schedule.Job
                A job without a function, used to specify the time interval.
            object_name (optional): str
                The name of the object containing in the 'func' var to be executed.
            receive_job (optional): bool
                A flag indicating whether the job should be received from an object from 'func' var.
            save (optional): bool
                A flag indicating whether the job should be saved.
            max_live (optional): bool
                A flag indicating whether the job should have a maximum live time.
            serializer (optional): bool
                json pickel or dill must have a dumps fuction
            *args, **kwargs (optional):
                Additional arguments to be passed to the job function.


        Parameters
            ----------
           job_data : dict

        example usage
            ----------
            `python

            `

    """
        if job_data is None:
            self.app.logger.error("No job data provided")
            return None
        job_id = job_data["job_id"]
        second = job_data.get("second", 0)
        func = job_data.get("func", None)
        job = job_data.get("job", None)
        time_passer = job_data.get("time_passer", None)
        object_name = job_data.get("object_name", "tb_job_fuction")
        receive_job = job_data.get("receive_job", False)
        save = job_data.get("save", False)
        max_live = job_data.get("max_live", True)
        serializer = job_data.get("serializer", serializer_default)
        args = job_data.get("args", ())
        kwargs = job_data.get("kwargs", {})

        return self.register_job(
            job_id=job_id,
            second=second,
            func=func,
            job=job,
            time_passer=time_passer,
            object_name=object_name,
            receive_job=receive_job,
            save=save,
            max_live=max_live,
            serializer=serializer,
            args=args,
            kwargs=kwargs
        )


if __name__ == '__main__':
    '''
    def example_basic():
        print("example")


    def example_args(test='default'):
        print("example args=", test)


    test_var_int = 0
    test_var_list = [0]
    test_var_dict = {"data": 0}


    def example_closer():
        print(f"data :\n\t{test_var_int=}\t{test_var_list=}\t{test_var_dict=}")


    from_string = """def example_basic():
    print("example_from_string")"""

    with open('example_file.py', 'w') as f1:
        f1.write(from_string.replace('_from_string', '_from_file'))


    def example_dill():
        print("example_dill")


    import dill

    with open('example_file.dill', 'wb') as f:
        dill.dump(example_dill, f)

    init_sm(get_app(name='debug'))  # or  from toolboxv2 import get_app, tbef ; app = get_app()
    register_instance(  # or app.run_any(tbef.SCHEDULER_MANAGER.ADD,
        job_data={
            "job_id": "job-example_basic",
            "second": 20,
            "func": example_basic,
            "job": None,
            "time_passer": None,
            "object_name": "tb_job_fuction",
            "receive_job": False,
            "save": False,
            "max_live": False
        })
    register_instance(job_data={
        "job_id": "job-example_args",
        "second": 10,
        "func": example_args,
        "job": None,
        "time_passer": None,
        "object_name": "tb_job_fuction",
        "receive_job": False,
        "save": False,
        "max_live": False,
        "args": ['update']
    })
    register_instance(job_data={
        "job_id": "job-example_closer",
        "second": 5,
        "func": example_closer,
        "job": None,
        "time_passer": None,
        "object_name": "tb_job_fuction",
        "receive_job": False,
        "save": False,
        "max_live": False
    })
    register_instance(job_data={
        "job_id": "job-from_string",
        "second": 25,
        "func": from_string,
        "job": None,
        "time_passer": None,
        "object_name": "example_basic",
        "receive_job": False,
        "save": False,
        "max_live": False
    })
    register_instance(job_data={
        "job_id": "job-example_file.dill",
        "second": 35,
        "func": "example_file.dill",
        "job": None,
        "time_passer": None,
        "object_name": "example_dill",
        "receive_job": False,
        "save": False,
        "max_live": False
    })
    register_instance(job_data={
        "job_id": "job-example_file.py",
        "second": 37,
        "func": "example_file.py",
        "job": None,
        "time_passer": None,
        "object_name": "example_basic",
        "receive_job": False,
        "save": False,
        "max_live": False
    })
    import schedule

    # >>> schedule.every(10).minutes
    # >>> schedule.every(5).to(10).days
    # >>> schedule.every().hour
    # >>> schedule.every().day.at("10:30")
    register_instance(job_data={
        "job_id": "job-example_basic-at-10",
        "second": 0,
        "func": example_args,
        "job": None,
        "time_passer": schedule.every().day.at("22:30"),
        "object_name": "tb_job_fuction",
        "receive_job": False,
        "save": False,
        "max_live": False,
        "args": (" at 22:04",)
    })

    time.sleep(15)
    print("SET DATA TO 1")

    test_var_int.__add__(1)
    test_var_list[0] = 1
    test_var_dict["data"] = 1

    time.sleep(60)
    on_exit(app=get_app(name='debug'))  # or  app.exit() # for clean up wen using the app
    """
Expected output :

Starting ToolBox as test from : \\ToolBoxV2\\toolboxv2
Logger in Test Mode
================================
2024-03-06 22:29:17 INFO - Logger initialized
2024-03-06 22:29:17 INFO - Starting Application instance
2024-03-06 22:29:17 INFO - loading test-DESKTOP-CI57V1L.config
2024-03-06 22:29:17 INFO - Opening file in mode : r+
2024-03-06 22:29:17 INFO - Collecting data from storage key : dev~mode~:
2024-03-06 22:29:17 INFO - Collecting data from storage key : provider::
2024-03-06 22:29:17 INFO - Finish init up in t-0.022403000009944662s
2024-03-06 22:29:17 INFO - App instance, returned ID: test-DESKTOP-CI57V1L
SYSTEM :: DESKTOP-CI57V1L
ID -> test-DESKTOP-CI57V1L,
Version -> 0.1.8,

STARTING SchedulerManager
Added Job job-example_basic : - kwargs 1
Added Job job-example_args : - args 1 - kwargs 1
Added Job job-example_closer : - kwargs 1
Added Job job-from_string : - kwargs 1
Added Job job-example_file.dill : - kwargs 1
Added Job job-example_file.py : - kwargs 1
Added Job job-example_basic-at-10 : - args 1 - kwargs 1

2024-03-06 22:29:22 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})
data :
	test_var_int=0	test_var_list=[0]	test_var_dict={'data': 0}

2024-03-06 22:29:27 DEBUG - Running job Job(interval=10, unit=seconds, do=example_args, args=('update',), kwargs={})
example args= update
...

SET DATA TO 1

2024-03-06 22:29:32 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})
data :
	test_var_int=0	test_var_list=[1]	test_var_dict={'data': 1}

...
2024-03-06 22:29:42 DEBUG - Running job Job(interval=25, unit=seconds, do=example_basic, args=(), kwargs={})
2024-03-06 22:29:42 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})

example args= update
data :
	test_var_int=0	test_var_list=[1]	test_var_dict={'data': 1}

2024-03-06 22:29:47 DEBUG - Running job Job(interval=10, unit=seconds, do=example_args, args=('update',), kwargs={})
2024-03-06 22:29:47 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})

example_dill
data :
	test_var_int=0	test_var_list=[1]	test_var_dict={'data': 1}

2024-03-06 22:29:52 DEBUG - Running job Job(interval=35, unit=seconds, do=example_dill, args=(), kwargs={})
2024-03-06 22:29:52 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})
2024-03-06 22:29:54 DEBUG - Running job Job(interval=37, unit=seconds, do=example_basic, args=(), kwargs={})

example_from_file
example
example args= update
data :
	test_var_int=0	test_var_list=[1]	test_var_dict={'data': 1}
...
2024-03-06 22:29:57 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})
2024-03-06 22:30:00 DEBUG - Running job Job(interval=1, unit=days, do=example_args, args=(' at 22:04',), kwargs={})

example args=  at 22:04
    # maby u wondering why the system time is 22:30:00 but the args for the timed functions is 22:04 '26mins debug time..'
data :
	test_var_int=0	test_var_list=[1]	test_var_dict={'data': 1}
2024-03-06 22:30:02 DEBUG - Running job Job(interval=5, unit=seconds, do=example_closer, args=(), kwargs={})
...
    """
'''
    pass
