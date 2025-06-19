import os
import pickle
from collections import deque
from collections.abc import Callable

try:
    import dill
except ImportError:
    dill = pickle

import threading
import time
from datetime import datetime, timedelta

import schedule

from toolboxv2 import MainTool, Result, get_app

Name = 'SchedulerManager'
export = get_app(Name).tb
version = '0.0.2'

safety_mode = ['open', 'strict', 'closed'][1]
serializer_default, deserializer_default = [(dill, dill), (dill, dill), (pickle, pickle)] \
    [['open', 'strict', 'closed'].index(safety_mode)]


class SchedulerManagerClass:
    def __init__(self):
        self.jobs = {}
        self.thread = None
        self.running = False
        self.last_successful_jobs = deque(maxlen=3)  # Stores last 3 successful job names
        self.job_errors = {}  # Stores job names as keys and error messages as values

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

    def job_wrapper(self, job_name: str, job_function: callable):
        """
        Wrap a job function to track success and errors.
        """
        def wrapped_job(*args, **kwargs):
            try:
                job_function(*args, **kwargs)
                # If the job ran successfully, store it in the success queue
                self.last_successful_jobs.append(job_name)
                if job_name in self.job_errors:
                    del self.job_errors[job_name]  # Remove error record if job succeeded after failing
            except Exception as e:
                # Capture any exceptions and store them
                self.job_errors[job_name] = str(e)

        return wrapped_job


    def register_job(self,
                     job_id: str,
                     second: int = -1,
                     func: (Callable or str) | None = None,
                     job: schedule.Job | None = None,
                     time_passer: schedule.Job | None = None,
                     object_name: str | None = None,
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
            return Result.default_internal_error("Both job and func are not specified."
                                                 " Please specify either job or func.")
        if job is not None and func is not None:
            return Result.default_internal_error("Both job and func are specified. Please specify either job or func.")

        if job is not None:
            def func(x):
                return x
            return self._save_job(job_id=job_id,
                                  job=job,
                                  save=save,
                                  func=func,
                                  args=args,
                                  kwargs=kwargs,
                                  serializer=serializer)

        parsed_attr = self._parse_function(func=func, object_name=object_name)

        if parsed_attr.is_error():
            parsed_attr.result.data_info = f"Error parsing function for job : {job_id}"
            return parsed_attr

        if receive_job:
            job = parsed_attr.get()
        else:
            func = parsed_attr.get()

        time_passer = self._prepare_time_passer(time_passer=time_passer,
                                                second=second)

        job_func = self._prepare_job_func(func=func,
                                          max_live=max_live,
                                          second=second,
                                          args=args,
                                          kwargs=kwargs,
                                          job_id=job_id)

        job = self._get_final_job(job=job,
                                  func=self.job_wrapper(job_id, job_func),
                                  time_passer=time_passer,
                                  job_func=job_func,
                                  args=args,
                                  kwargs=kwargs)
        if job.is_error():
            return job

        job = job.get()

        return self._save_job(job_id=job_id,
                              job=job,
                              save=save,
                              func=func,
                              args=args,
                              kwargs=kwargs,
                              serializer=serializer)

    @staticmethod
    def _parse_function(func: str or Callable, object_name):
        if isinstance(func, str) and func.endswith('.py'):
            with open(func) as file:
                func_code = file.read()
                exec(func_code)
                func = locals()[object_name]
        elif isinstance(func, str) and func.endswith('.dill') and safety_mode == 'open':
            try:
                with open(func, 'rb') as file:
                    func = dill.load(file)
            except FileNotFoundError:
                return Result.default_internal_error(f"Function file {func} not found or dill not installed")
        elif isinstance(func, str):
            local_vars = {'app': get_app(from_=Name + f".pasing.{object_name}")}
            try:
                exec(func.strip(), {}, local_vars)
            except Exception as e:
                return Result.default_internal_error(f"Function parsing failed withe {e}")
            func = local_vars[object_name]
        elif isinstance(func, Callable):
            pass
        else:
            return Result.default_internal_error("Could not parse object scheduler_manager.parse_function")
        return Result.ok(func)

    @staticmethod
    def _prepare_time_passer(time_passer, second):
        if time_passer is None and second > 0:
            return schedule.every(second).seconds
        elif time_passer is None and second <= 0:
            raise ValueError("second must be greater than 0")
        return time_passer

    def _prepare_job_func(self, func: Callable, max_live: bool, second: float, job_id: str, *args, **kwargs):
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
            return Result.default_internal_error("No Final job found for register")
        return Result.ok(job)

    def _save_job(self, job_id, job, save, args=None, **kwargs):
        if job is not None:
            self.jobs[job_id] = {'id': job_id, 'job': job, 'save': save, 'func': job_id, 'args': args,
                                 'kwargs': kwargs}
            f = (f"Added Job {job_id} :{' - saved' if save else ''}"
                  f"{' - args ' + str(len(args)) if args else ''}"
                  f"{' - kwargs ' + str(len(kwargs.keys())) if kwargs else ''}")
            return Result.ok(f)
        else:
            return Result.default_internal_error(job_id)

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

    def get_tasks_table(self):
        if not self.jobs:
            return "No tasks registered."

        # Calculate the maximum width for each column
        id_width = max(len("Task ID"), max(len(job_id) for job_id in self.jobs))
        next_run_width = len("Next Execution")
        interval_width = len("Interval")

        # Create the header
        header = f"| {'Task ID':<{id_width}} | {'Next Execution':<{next_run_width}} | {'Interval':<{interval_width}} |"
        separator = f"|{'-' * (id_width + 2)}|{'-' * (next_run_width + 2)}|{'-' * (interval_width + 2)}|"

        # Create the table rows
        rows = []
        for job_id, job_info in self.jobs.items():
            job = job_info['job']
            next_run = job.next_run.strftime("%Y-%m-%d %H:%M:%S") if job.next_run else "N/A"
            interval = self._get_interval_str(job)
            row = f"| {job_id:<{id_width}} | {next_run:<{next_run_width}} | {interval:<{interval_width}} |"
            rows.append(row)

        # Combine all parts of the table
        table = "\n".join([header, separator] + rows)
        return table

    def _get_interval_str(self, job):
        if job.interval == 0:
            return "Once"

        units = [
            (86400, "day"),
            (3600, "hour"),
            (60, "minute"),
            (1, "second")
        ]

        for seconds, unit in units:
            if job.interval % seconds == 0:
                count = job.interval // seconds
                return f"Every {count} {unit}{'s' if count > 1 else ''}"

        return f"Every {job.interval} seconds"

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
        func = job_data.get("func")
        job = job_data.get("job")
        time_passer = job_data.get("time_passer")
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


def example_basic():
    print("example")


def example_args(test='default'):
    print("example args=", test)


test_var_int = 0
test_var_list = [0]
test_var_dict = {"data": 0}


def example_closer():
    print(f"data :\n\t{test_var_int=}\t{test_var_list=}\t{test_var_dict=}")


@export(test_only=True, mod_name=Name)
def test_scheduler():
    from_string = """def example_basic_S():
    print("example_from_string")"""

    with open('example_file.py', 'w') as f1:
        f1.write(from_string.replace('_from_string', '_from_file').replace('_S', '_D'))

    # def example_dill():
    #     print("example_dill")

    # import dill
    #
    # with open('example_file.dill', 'wb') as f:
    #     dill.dump(example_dill, f)

    # or  from toolboxv2 import get_app, TBEF ;
    app = get_app(name='debug')
    shm = app.save_load(Name)
    shm.init_sm()

    assert not shm.register_instance(  # or app.run_any(TBEF.SCHEDULER_MANAGER.ADD,
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
        }).print().is_error()
    assert not shm.register_instance(job_data={
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
    }).print().is_error()
    assert not shm.register_instance(job_data={
        "job_id": "job-example_closer",
        "second": 5,
        "func": example_closer,
        "job": None,
        "time_passer": None,
        "object_name": "tb_job_fuction",
        "receive_job": False,
        "save": False,
        "max_live": False
    }).print().is_error()

    # shm.register_instance(job_data={
    #     "job_id": "job-from_string",
    #     "second": 25,
    #     "func": from_string,
    #     "job": None,
    #     "time_passer": None,
    #     "object_name": "example_basic_S",
    #     "receive_job": False,
    #     "save": False,
    #     "max_live": False
    # }) not in same file possible
    # shm.register_instance(job_data={
    #     "job_id": "job-example_file.dill",
    #     "second": 35,
    #     "func": "example_file.dill",
    #     "job": None,
    #     "time_passer": None,
    #     "object_name": "example_dill",
    #     "receive_job": False,
    #     "save": False,
    #     "max_live": False
    # }) unsafe inport dill extra
    assert not shm.register_instance(job_data={
        "job_id": "job-example_file.py",
        "second": 2,
        "func": "example_file.py",
        "job": None,
        "time_passer": None,
        "object_name": "example_basic_D",
        "receive_job": False,
        "save": False,
        "max_live": False
    }).print().is_error()
    import schedule

    # >>> schedule.every(10).minutes
    # >>> schedule.every(5).to(10).days
    # >>> schedule.every().hour
    # >>> schedule.every().day.at("10:30")
    assert not shm.register_instance(job_data={
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
    }).print().is_error()

    time.sleep(15)
    print("SET DATA TO 1")

    test_var_int.__add__(1)
    test_var_list[0] = 1
    test_var_dict["data"] = 1

    time.sleep(15)  # or  app.exit() # for clean up wen using the app


if __name__ == '__main__':
    test_scheduler()
