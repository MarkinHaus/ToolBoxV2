import os.path

from starlette.responses import FileResponse, HTMLResponse, RedirectResponse

from toolboxv2 import App, get_app
from toolboxv2.mods.TruthSeeker.arXivCrawler import ArXivPDFProcessor

# from toolboxv2.mods.TruthSeeker.nGui import create_ui
from toolboxv2.utils.system.session import RequestSession

Name = 'TruthSeeker'
export = get_app("TruthSeeker.Export").tb
version = '0.0.1'
default_test = export(mod_name=Name, test_only=True, version=version)

NAME = "TruthSeeker"
dot = os.path.dirname(os.path.abspath(__file__))
content = open(os.path.join(dot,"template.html"), encoding="utf-8").read()
abut_content = open(os.path.join(dot,"abut.html"), encoding="utf-8").read()

code_templates = {
    "ontime":None,
    "48H": 48*1200,
    "1J":24*1200*366,

    "ADMIN":24*1200*366,
    "HELPERS":12*1200*366,

    "PROCESS":12*1200,
}
promo_templates = {
    "Promo15W":24*1200*7,
    "Promo15":None,
    "Promo25":None,
    "Promo50":None,
}

@export(mod_name=Name, version=version, initial=True)
def start(app=None):
    global talk_generate, talk_tts
    if app is None:
        app = get_app("Starting Talk interface")
    app.get_mod("isaa").load_keys_from_env()# (build=False)

    app.get_mod("CodeVerification")
    # print("before miain")
    app.run_any(("CodeVerification", "init_scope"), scope=Name)
    # print("before miain2")
    app.run_any(("CodeVerification", "init_scope"), scope=Name+'-promo')
    # print("After main12")
    # app.run_any(("CodeVerification", "reset_templates"), scope=Name)
    # app.run_any(("CodeVerification", "reset_templates"), scope=Name)
    code_templates_dict = app.run_any(("CodeVerification", "all_templates"), scope=Name)
    code_templates_names = [e['name'] for e in code_templates_dict.values() if 'name' in e]
    if len(code_templates.keys()) != len(code_templates_names):
        # app.run_any(("CodeVerification", "reset_templates"), scope=Name)
        for n, v in code_templates.items():
            if n in code_templates_names:
                continue
            app.run_any(("CodeVerification", "add_template"),
                        scope=Name,
                        name=n,
                        usage_type="one_time" if v is None else 'timed',
                        max_uses=1 if v is None else int(v/60*5),
                        valid_duration=v)

    # app.run_any(("CodeVerification", "reset_templates"), scope=Name + '-promo')
    promo_templates_dict = app.run_any(("CodeVerification", "all_templates"), scope=Name + '-promo')
    promo_templates_names = [e['name'] for e in promo_templates_dict.values() if 'name' in e]
    if len(promo_templates.keys()) != len(promo_templates_names):
        # app.run_any(("CodeVerification", "reset_templates"), scope=Name + '-promo')
        for n,v in promo_templates.items():
            if n in promo_templates_names:
                continue
            app.run_any(("CodeVerification", "add_template"),
                        scope=Name + '-promo',
                        name=n,
                        usage_type="one_time" if v is None else 'timed',
                        max_uses=1 if v is None or v == "PROCESS" else int(v/60*5),
                        valid_duration=v)
    # create_ui(ArXivPDFProcessor)
    #app.run_any(("CloudM","add_ui"),
    #            name="TruthSeeker",
    #            titel="TruthSeeker Research",
    #            path=f"/api/{MOD_NAME}/get_main_ui",
    #            description="AI Research Assistant"
    #            )
    print("DONE TruthSeeker")


@export(mod_name=Name, version=version, request_as_kwarg=True, level=-1, api=True,
        name="byCode", row=True)
def byCode(app = None, request: RequestSession or None = None):
    if app is None:
        app = get_app(f"{Name}.byCode")
    if request is None:
        return
    payKey, codeClass, ontimeKey = request.json()
    return {'code':"code"}


@export(mod_name=Name, version=version, level=-1, api=True,
        name="video", row=True)
def video(app = None):
    return FileResponse(r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\TruthSeeker\demo-demo.mp4")

@export(mod_name=Name, version=version, request_as_kwarg=True, level=-1, api=True,
        name="codes", row=True)
def codes(app = None, request: RequestSession or None = None):
    if app is None:
        app = get_app(f"{Name}.code")
    if request is None:
        return
    data = request.json()
    print(data)
    def timeEstimations(x):
        return [f'< {x}min', f'= {x}min', f'> {x}min']
    query, depth, promoCode, ontimeCode = data.get('query'), data.get('depth'), data.get('promoCode'), data.get('ontimeCode')
    promo = 1
    depth = depth.upper()
    price = 0 + len(query)//250 + [0, 250, 1200, 1500, 50][["Q","I", "E", "P", depth[0]].index(depth[0])]
    if promoCode:
        data_promo = app.run_any(("CodeVerification", "validate"), scope=Name + '-promo', code=promoCode)
        error = False
        if data_promo is None:
            error = True
        if not error and not data_promo.get('template_name', '').startswith("Promo"):
            promo = 1.5
        if not error and data_promo.get('template_name', '').startswith("Promo15"):
            promo = .85
        if not error and data_promo.get('template_name', '').startswith("Promo25"):
            promo = .75
        if not error and data_promo.get('template_name', '').startswith("Promo50"):
            promo = .5
    price *= promo
    if price != 0:
        if ontimeCode is None:
            return {'ontimeKey': 'None', 'valid': False, 'ppc': {'timeEstimation': '', 'price': min(2000, max(price, 0))}}
        data_ontime = app.run_any(("CodeVerification", "validate"), scope=Name, code=ontimeCode)
        if data_ontime is None:
            return {'ontimeKey':'None', 'valid':False, 'ppc': {'timeEstimation': 'Invalid Ontime Code','price': min(2000, max(price, 0))}}

    ontimeKey = app.run_any(("CodeVerification", "generate"), scope=Name, template_id="PROCESS")
    return {'ontimeKey':ontimeKey, 'valid':True,
            'ppc':
                {'timeEstimation': timeEstimations(((price+100)*2)//100)[0 if (price+1)//100 < 5 else (1 if (price+1)//100 ==5 else 2)],
                 'price': min(2000, max(price, 0))}}

@export(mod_name=Name, version=version, level=1, api=False,
        name="process_cli")
def process_cli(app = None, *query: str or None):
    if app is None:
        app = get_app(f"{Name}.process")
    if query is None:
        return
    return ArXivPDFProcessor(' '.join(query),tools=app.get_mod("isaa"), max_workers=5).process()

@export(mod_name=Name, version=version, request_as_kwarg=True, level=-1, api=True,
        name="process", row=True)
def process(app = None, request: RequestSession or None = None):
    if app is None:
        app = get_app(f"{Name}.process")
    if request is None:
        return

    data =  request.json()

    print(data)

    query, depth, ontimeKey, _email =  data.get('query', ''), data.get('depth', ''), data.get('ontimeKey', ''), data.get('email', '')
    data_key = app.run_any(("CodeVerification", "validate"), scope=Name, code=ontimeKey)
    error = False
    if data_key is None:
        error = True
    if not error and data_key.get('template_name') != 'PROCESS':
        error = True
    if not error and data_key.get('usage_type') != 'timed':
        error = True
    if not error and data_key.get('uses_count') != 1:
        error = True
    if error:
       return {'is_true': str(data_key), 'summary':"INVALID QUERY", 'insights': [], 'papers':[]}
    depth = depth.upper()
    config = [{
        "chunk_size":  5000,
        "overlap":   500,
        "limiter" : 0.04,
        "max_workers" : 6,
        "num_search_result_per_query" : 3,
        "max_search" : 5,
    },{
        "chunk_size":  2000,
        "overlap":   200,
        "limiter" : 0.1,
        "max_workers" : 6,
        "num_search_result_per_query" : 4,
        "max_search" : 6,
    },{
        "chunk_size":  600,
        "overlap":   20,
        "limiter" : 0.4,
        "max_workers" : 12,
        "num_search_result_per_query" : 5,
        "max_search" : 10,
    },{
        "chunk_size":  1000,
        "overlap":   100,
        "limiter" : 0.24,
        "max_workers" : None,
        "num_search_result_per_query" : 5,
        "max_search" : 8,
    },{
        "chunk_size":  1000,
        "overlap":   200,
        "limiter" : 0.51,
        "max_workers" : 1,
        "num_search_result_per_query" : 5,
        "max_search" : 4,
    },][["Q", "I", "E", "P", depth[0]].index(depth[0])]
    papers, insights = ArXivPDFProcessor(query,tools=app.get_mod("isaa"), **config).process()
    return {'is_true': insights.is_true, 'summary':insights.summary, 'insights': insights.key_point.split(">\n\n<"), 'papers':[p.model_dump() for p in papers]}

@export(mod_name=Name, version=version, api=True,
        name="main_web_entry", row=True,request_as_kwarg=True)
def main_web_entry(app: App = None, abut=None, request: RequestSession or None = None):
    if abut:
        return HTMLResponse(content=abut_content)
    if hasattr(request, 'row'):
        if sid := request.row.query_params.get('session_id'):
            return RedirectResponse(url=f"/gui/open-Seeker.seek?session_id={sid}")
    if hasattr(request, 'query_params'):
        if sid := request.query_params.get('session_id'):
            return RedirectResponse(url=f"/gui/open-Seeker.seek?session_id={sid}")
    return RedirectResponse(url="/gui/open-Seeker")

