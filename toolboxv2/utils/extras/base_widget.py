import uuid

from ..system.all_functions_enums import CLOUDM_AUTHMANAGER

try:
    from ..system.all_functions_enums import MINIMALHTML
except ImportError:
    MINIMALHTML = lambda: None
    MINIMALHTML.ADD_GROUP = ("MINIMALHTML", "ADD_GROUP".lower())
    MINIMALHTML.GENERATE_HTML = ("MINIMALHTML", "GENERATE_HTML".lower())
    MINIMALHTML.ADD_COLLECTION_TO_GROUP = ("MINIMALHTML", "ADD_COLLECTION_TO_GROUP".lower())


class BaseWidget:
    def __init__(self, name: str):
        self.name = name
        self.openWidgetsIDs = {}
        self.onReload = []

    def register(self, app, fuction, version=None):
        if version is None:
            version = app.version
        app.tb(mod_name=self.name, version=version, request_as_kwarg=True, level=1, api=True, name="get_widget")(
            fuction)

    def modify_iterator(self, iterator, replace):
        """
        ['a', 'b'] -> [{replace[0]: 'a',..., replace[len(replace)-1]: 'a'},
        {replace[0]: 'b',..., replace[len(replace)-1]: 'b'}, ]
        """

        for item in iterator:
            modified_item = {replace[i]: (self.name if replace[i] == "name" else '') + item for i in
                             range(len(replace))}
            yield modified_item

    def register2reload(self, *functions):
        for fuction in functions:
            x = lambda r: fuction(request=r)
            self.onReload.append(x)

    @staticmethod
    def get_a_group(asset_name, template=None, file_path=None, a_kwargs=None):
        if a_kwargs is None:
            raise ValueError("a_kwargs must be specified")
        return [{'name': asset_name,
                 'file_path': file_path,
                 'kwargs': a_kwargs
                 } if file_path is not None else {'name': asset_name,
                                                  'template': template,
                                                  'kwargs': a_kwargs
                                                  }]

    def group_generator(self, asset_name: str, iterator: iter, template=None, file_path=None, a_kwargs=None):
        groups = []
        work_kwargs = a_kwargs
        for data in enumerate(iterator):
            if isinstance(data, dict):
                work_kwargs = {**a_kwargs, **data}
            groups.append(self.get_a_group(asset_name, template=template, file_path=file_path, a_kwargs=work_kwargs))
        return groups

    def asset_loder(self, app, name, asset_id, file_path=None, template=None, iterator=None, **kwargs):
        a_kwargs = {**{
            'root': f"/api/{self.name}",
            'WidgetID': asset_id},
                    **kwargs}
        asset_name = f"{name}{asset_id}"
        if iterator is None:
            group = self.get_a_group(asset_name,
                                     template=template,
                                     file_path=file_path,
                                     a_kwargs=a_kwargs)
        else:
            group = self.group_generator(asset_name,
                                         iterator=iterator,
                                         template=template,
                                         file_path=file_path,
                                         a_kwargs=a_kwargs)

        asset = app.run_any(MINIMALHTML.ADD_COLLECTION_TO_GROUP,
                            group_name=self.name,
                            collection={'name': f"{self.name}-{asset_name}",
                                        'group': group},
                            get_results=True)
        if asset.is_error():
            app.run_any(MINIMALHTML.ADD_GROUP, command=self.name)
            asset = app.run_any(MINIMALHTML.ADD_COLLECTION_TO_GROUP,
                                group_name=self.name,
                                collection={'name': f"{self.name}-{asset_name}",
                                            'group': group},
                                get_results=True)
        return asset

    def generate_html(self, app, name="MainWidget", asset_id=str(uuid.uuid4())[:4]):
        return app.run_any(MINIMALHTML.GENERATE_HTML,
                           group_name=self.name,
                           collection_name=f"{name}{asset_id}")

    def load_widget(self, app, request, name="MainWidget", asset_id=str(uuid.uuid4())[:4]):
        app.run_any(MINIMALHTML.ADD_GROUP, command=self.name)
        self.reload(request)
        html_widget = self.generate_html(app, name, asset_id)
        return html_widget[0]['html_element']

    @staticmethod
    async def get_user_from_request(app, request):
        from toolboxv2.mods.CloudM import User
        name = request.session.get('live_data', {}).get('user_name', "Cud be ur name")
        if name != "Cud be ur name":
            user = await app.a_run_any(CLOUDM_AUTHMANAGER.GET_USER_BY_NAME,
                                       username=app.config_fh.decode_code(name))
        else:
            user = User()
        return user

    @staticmethod
    def get_s_id(request):
        from ..system.types import Result
        if request is None:
            return Result.default_internal_error("No request specified")
        return Result.ok(request.session.get('ID', ''))

    def reload(self, request):
        [_(request) for _ in self.onReload]

    async def get_widget(self, request):
        raise NotImplementedError
