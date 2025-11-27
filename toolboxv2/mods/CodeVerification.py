import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from toolboxv2 import Result, get_app

Name = 'CodeVerification'
version = '0.0.1'
export = get_app(f"{Name}.Export").tb
closed_api_export = export(mod_name=Name, version=version, level=1, test=False)
open_api_export = export(mod_name=Name, version=version, level=0, api=True, test=False)


@dataclass
class ConfigTemplate:
    name: str
    usage_type: str  # 'one_time' or 'timed'
    max_uses: int = 1
    valid_duration: int | None = None  # in seconds
    additional_params: dict[str, Any] = field(default_factory=dict)
    scope: str = "main"


class VerificationSystem:
    def __init__(self, tools_db, scope="main"):
        """
        Initialize VerificationSystem with DB Tools integration

        Args:
            tools_db (Tools): Database tools from toolboxv2.mods.DB
            scope (str, optional): Scope for templates and codes. Defaults to "main".
        """
        self.tools_db = tools_db
        self.scope = scope
        self.tidmp = {}
        self._ensure_scope_templates()

    def get(self):
        return self

    def reset_scope_templates(self):
        """
        Ensure a templates dictionary exists for the current scope in the database
        """
        templates_key = f"verification_templates_{self.scope}"

        self.tools_db.set(templates_key, json.dumps({}))

    def _ensure_scope_templates(self):
        """
        Ensure a templates dictionary exists for the current scope in the database
        """
        templates_key = f"verification_templates_{self.scope}"

        # Check if templates exist for this scope
        templates_exist = self.tools_db.if_exist(templates_key)

        if templates_exist.is_error() and not templates_exist.is_data():
            # Initialize empty templates dictionary if not exists
            self.tools_db.set(templates_key, json.dumps({}))
        else:
            allt = self.get_all_templates()

            for k, v in allt.items():
                if 'name' not in v:
                    continue
                self.tidmp[v['name']] = k

    def add_config_template(self, template: ConfigTemplate) -> str:
        """
        Add a new configuration template to the database

        Args:
            template (ConfigTemplate): The configuration template

        Returns:
            str: Unique identifier of the template
        """
        # Ensure template has the current scope
        template.scope = self.scope

        # Generate a unique template ID
        template_id = secrets.token_urlsafe(8)

        # Get existing templates for this scope
        templates = self.get_all_templates()

        # Add new template
        self.tidmp[template.name] = template_id
        templates[template_id] = asdict(template)

        # Save updated templates back to database
        templates_key = f"verification_templates_{self.scope}"
        save_result = self.tools_db.set(templates_key, json.dumps(templates))

        if save_result.is_error():
            raise ValueError("Could not save template")

        return template_id

    def get_all_templates(self):
        templates_key = f"verification_templates_{self.scope}"
        templates_result = self.tools_db.get(templates_key)

        if not templates_result.is_error() and templates_result.is_data():
            try:
                templates_result.result.data = json.loads(templates_result.get())
            except Exception as e:
                templates_result.print()
                print(f"Errro loding template data curupted : {str(e)}")
                templates_result.result.data = {}
        else:
            templates_result.result.data = {}
        if not isinstance(templates_result, dict):
            templates_result = templates_result.result.data
        return templates_result

    def generate_code(self, template_id: str) -> str:
        """
        Generate a code based on the configuration template

        Args:
            template_id (str): ID of the configuration template

        Returns:
            str: Generated verification code
        """
        # Get templates for this scope
        templates = self.get_all_templates()
        print(templates, self.tidmp, template_id)
        if template_id not in templates:
            template_id = self.tidmp.get(template_id, template_id)
        if template_id not in templates:
            raise ValueError("Invalid configuration template")

        template_dict = templates[template_id]
        ConfigTemplate(**template_dict)

        # Generate a random code with max 16 characters
        code = secrets.token_urlsafe(10)[:16]

        # Prepare code information
        code_info = {
            'template_id': template_id,
            'created_at': time.time(),
            'uses_count': 0,
            'scope': self.scope
        }

        # Store code information in database
        codes_key = f"verification_codes_{self.scope}"
        existing_codes_result = self.tools_db.get(codes_key)

        existing_codes = {}
        if not existing_codes_result.is_error() and existing_codes_result.is_data():
            d = existing_codes_result.get()
            if isinstance(d, list):
                d = d[0]
            existing_codes = json.loads(d)

        existing_codes[code] = code_info

        save_result = self.tools_db.set(codes_key, json.dumps(existing_codes))

        if save_result.is_error():
            raise ValueError("Could not save generated code")

        return code

    def validate_code(self, code: str) -> dict[str, Any] | None:
        """
        Validate a code and return template information

        Args:
            code (str): Code to validate

        Returns:
            Optional[Dict[str, Any]]: Template information for valid code, else None
        """
        # Get codes for this scope
        codes_key = f"verification_codes_{self.scope}"
        codes_result = self.tools_db.get(codes_key)

        if codes_result.is_error() or not codes_result.is_data():
            return None

        d = codes_result.get()
        if isinstance(d, list):
            d = d[0]
        existing_codes = json.loads(d)

        if code not in existing_codes:
            return None

        code_info = existing_codes[code]

        # Check if code is from the same scope
        if code_info.get('scope') != self.scope:
            return None

        # Get templates for this scope
        templates = self.get_all_templates()
        template_id = code_info['template_id']

        if template_id not in templates:
            return templates

        template_dict = templates[template_id]
        template = ConfigTemplate(**template_dict)

        # Check usage count
        if code_info['uses_count'] >= template.max_uses:
            del existing_codes[code]
            self.tools_db.set(codes_key, json.dumps(existing_codes))
            return None

        # Check time validity for timed codes
        if template.usage_type == 'timed':
            current_time = time.time()
            if template.valid_duration and (current_time - code_info['created_at']) > template.valid_duration:
                del existing_codes[code]
                self.tools_db.set(codes_key, json.dumps(existing_codes))
                return None

        # Update uses count
        existing_codes[code]['uses_count'] += 1
        uses_count = existing_codes[code].get('uses_count', 1)
        # Remove code if it's a one-time use
        if template.usage_type == 'one_time':
            del existing_codes[code]

        # Save updated codes
        self.tools_db.set(codes_key, json.dumps(existing_codes))

        return {
            'template_name': template.name,
            'usage_type': template.usage_type,
            'uses_count': uses_count
        }


# Example usage function
@export(mod_name=Name, version=version, level=0, test_only=True)
def example_usage(app=None):
    # Create verification systems with different scopes
    tools_db = app.get_mod("DB")
    tools_db.edit_cli("LC")
    vs_main = VerificationSystem(tools_db, scope="test-main")
    vs_secondary = VerificationSystem(tools_db, scope="test-main")
    vs_secondary2 = VerificationSystem(tools_db, scope="test-main2")

    # Create templates
    one_time_template = ConfigTemplate(
        name="One-time Registration",
        usage_type="one_time",
        max_uses=2
    )

    timed_template = ConfigTemplate(
        name="24-Hour Access",
        usage_type="timed",
        max_uses=99999999,
        valid_duration=24 * 60 * 60  # 24 hours
    )

    # Add templates to different scopes
    one_time_id_main = vs_main.add_config_template(one_time_template)
    timed_id_main = vs_main.add_config_template(timed_template)

    # Generate codes
    one_time_code_main = vs_main.generate_code(one_time_id_main)
    one_time_code_main2 = vs_main.generate_code(one_time_id_main)
    timed_code_main = vs_main.generate_code(timed_id_main)

    # Validate codes (demonstrating scope isolation)
    print("Main Scope Validations:")
    assert vs_main.validate_code(timed_code_main) is not None
    assert vs_main.validate_code(timed_code_main) is not None
    assert vs_main.validate_code(one_time_code_main) is not None
    assert vs_main.validate_code(one_time_code_main) is None
    assert vs_main.validate_code(one_time_code_main) is None

    print("\nSecondary Scope Validations:")
    assert vs_secondary.validate_code(timed_code_main) is not None
    assert vs_secondary.validate_code(one_time_code_main2) is not None
    assert vs_secondary.validate_code(one_time_code_main2) is None

    assert vs_secondary2.validate_code(timed_code_main) is not None

    assert vs_main.get_all_templates() == vs_secondary.get_all_templates()
    assert vs_secondary.get_all_templates() == vs_secondary2.get_all_templates()
    assert len(vs_main.get_all_templates().keys()) != 0
    assert len(vs_main.get_all_templates().keys()) == 2


# Note: This requires a tools_db instance from toolboxv2.mods.DB to be passed in


if __name__ == "__main__":
    example_usage[0](get_app())

VS = {}


@closed_api_export
def init_scope(app=None, scope="Main"):
    if app is None:
        app = get_app(Name)
    if scope in VS:
        return VS[scope]
    tools = app.get_mod("DB", spec=scope)
    if tools.mode.value != "RR":
        tools.edit_cli("RR")
    vs = VerificationSystem(tools_db=tools, scope=scope)
    VS[scope] = vs
    return Result.ok(vs)


@closed_api_export
def add_template(app=None, scope="Main", name="null",
                 usage_type="null",
                 max_uses=1,
                 valid_duration=None, **kwargs):
    if app is None:
        app = get_app(Name)

    # Add templates to different scopes
    return Result.ok(init_scope(app, scope).get().add_config_template(ConfigTemplate(
        name=name,
        usage_type=usage_type,
        max_uses=max_uses,
        valid_duration=valid_duration,  # 24 hours
        additional_params=kwargs,
    )))


@export(mod_name=Name, version=version, level=1, api=True, name="generate_api")
@closed_api_export
def generate(app=None, scope="Main", template_id=None):
    if app is None:
        app = get_app(Name)
    if template_id is None:
        return ""

    # Add templates to different scopes
    return Result.ok(init_scope(app, scope).get().generate_code(template_id))


@export(mod_name=Name, version=version, level=0, name="validate")
@open_api_export
def validate_api(app=None, scope="Main", code=None):
    if app is None:
        app = get_app(Name)
    if code is None:
        return ""

    return Result.ok(init_scope(app, scope).get().validate_code(code))


@closed_api_export
def all_templates(app=None, scope="Main"):
    if app is None:
        app = get_app(Name)

    return Result.ok(init_scope(app, scope).get().get_all_templates())


@closed_api_export
def reset_templates(app=None, scope="Main"):
    if app is None:
        app = get_app(Name)

    # Add templates to different scopes
    return Result.ok(init_scope(app, scope).get().reset_scope_templates())


from toolboxv2.utils.extras.base_widget import get_user_from_request

withe_list = ['root', 'loot']
scoped_withe_list = {}


@export(mod_name=Name, version=version, api=True,
        name="pannel", row=True, request_as_kwarg=True)
async def pannel(app=None, request=None, scope="Main"):
    if request is None:
        return Result.html("<h1>No access</h1><a href='/'>home</a>")
    if app is None:
        app = get_app(Name)

    user = await get_user_from_request(app, request)
    if user.name not in withe_list:
        return Result.html("<h1>No access not on with list </h1><a href='/'>home</a>")
    return Result.html(html_template)


"""if __name__ == "__main__":

    get_app().get_mod("DB").edit_cli("LC")
    example_usage(get_app().get_mod("DB"))"""

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Code Verification Admin Dashboard</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --background-color: #ecf0f1;
            --text-color: #333;
            --white: #ffffff;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }

        body {
            background-color: var(--background-color);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .card {
            background-color: var(--white);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }

        .form-control {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        .btn {
            display: inline-block;
            padding: 10px 15px;
            background-color: var(--secondary-color);
            color: var(--white);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .btn:hover {
            background-color: #2980b9;
        }

        .btn-block {
            width: 100%;
        }

        .template-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .template-button {
            padding: 10px;
            background-color: var(--primary-color);
            color: var(--white);
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        .generated-code {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            word-break: break-all;
        }

        .hidden {
            display: none;
        }

        .templates-table {
            width: 100%;
            border-collapse: collapse;
        }

        .templates-table th,
        .templates-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .templates-table th {
            background-color: var(--primary-color);
            color: var(--white);
        }

        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center; margin-bottom: 20px;">Code Verification Admin Dashboard</h1>

        <div class="dashboard">
            <!-- Template Creation Section -->
            <div class="card">
                <h2>Create New Template</h2>
                <form id="templateForm">
                    <div class="form-group">
                        <label>Template Name</label>
                        <input type="text" id="templateName" class="form-control" required>
                    </div>
                    <div class="form-group">
                        <label>Usage Type</label>
                        <select id="usageType" class="form-control">
                            <option value="one_time">One-Time Use</option>
                            <option value="timed">Timed Use</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Max Uses</label>
                        <input type="number" id="maxUses" class="form-control" min="1" value="1">
                    </div>
                    <div id="durationContainer" class="form-group hidden">
                        <label>Valid Duration (seconds)</label>
                        <input type="number" id="validDuration" class="form-control" min="0">
                    </div>
                    <button type="submit" class="btn btn-block">Create Template</button>
                </form>
            </div>

            <!-- Code Generation Section -->
            <div class="card">
                <h2>Generate Verification Code</h2>
                <div id="templateList" class="template-list">
                    <!-- Templates will be dynamically populated here -->
                </div>
                <div id="generatedCodeArea" class="hidden">
                    <div id="generatedCodeDisplay" class="generated-code"></div>
                </div>
            </div>

            <!-- Existing Templates Section -->
            <div class="card" style="grid-column: 1 / -1;">
                <h2>Existing Templates</h2>
                <table class="templates-table">
                    <thead>
                        <tr>
                            <th>Scope</th>
                            <th>Name</th>
                            <th>Usage Type</th>
                            <th>Max Uses</th>
                            <th>Valid Duration</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="templatesTableBody">
                        <!-- Templates will be dynamically populated here -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // API Interaction Functions
        const API_BASE = '/api/CodeVerification/';
        const params = new URLSearchParams(window.location.search);
        const scope = params.get('scope')

        async function fetchData(endpoint, method = 'GET', body = null) {
            try {
                const options = {
                    method,
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: body ? JSON.stringify(body) : null
                };
                const response = await fetch(API_BASE + endpoint, options);
                return await response.json();
            } catch (error) {
                console.error('API Error:', error);
                alert('An error occurred while communicating with the server');
            }
        }

        // Event Listeners and Page Logic
        document.getElementById('usageType').addEventListener('change', (e) => {
            const durationContainer = document.getElementById('durationContainer');
            durationContainer.classList.toggle('hidden', e.target.value !== 'timed');
        });

        document.getElementById('templateForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const templateName = document.getElementById('templateName').value;
            const usageType = document.getElementById('usageType').value;
            const maxUses = document.getElementById('maxUses').value;
            const validDuration = document.getElementById('validDuration').value || null;

            try {
                const result = await fetchData(`add_template?scope=${scope}`, 'POST', {
                    scope,
                    name: templateName,
                    usage_type: usageType,
                    max_uses: parseInt(maxUses),
                    valid_duration: usageType === 'timed' ? parseInt(validDuration) : null
                });
                alert(`Template created with ID: ${result}`);
                loadTemplates();
            } catch (error) {
                console.error('Error creating template:', error);
            }
        });

        async function loadTemplates() {
            const templateList = document.getElementById('templateList');
            const templatesTableBody = document.getElementById('templatesTableBody');

            templateList.innerHTML = '';
            templatesTableBody.innerHTML = '';

            try {
                let templates = await fetchData(`all_templates?scope=${scope}`);
                templates = JSON.parse(templates);
                templates = templates.result.data
                Object.entries(templates).forEach(([templateId, template]) => {
                    // Generate Code Buttons
                    const templateButton = document.createElement('button');
                    templateButton.textContent = `${template.name} (${template.scope})`;
                    templateButton.className = 'template-button';
                    templateButton.addEventListener('click', async () => {
                        try {
                            const generatedCode = await fetchData(`generate_api?scope=${template.scope}&template_id=${templateId}`, 'GET');
                            document.getElementById('generatedCodeDisplay').textContent = generatedCode;
                            document.getElementById('generatedCodeArea').classList.remove('hidden');
                        } catch (error) {
                            console.error('Error generating code:', error);
                        }
                    });
                    templateList.appendChild(templateButton);

                    // Templates Table
                    const row = templatesTableBody.insertRow();
                    row.innerHTML = `
                        <td>${template.scope}</td>
                        <td>${template.name}</td>
                        <td>${template.usage_type}</td>
                        <td>${template.max_uses}</td>
                        <td>${template.valid_duration ? template.valid_duration + 's' : 'N/A'}</td>
                        <td>
                            <button class="btn" onclick="generateCodeForTemplate('${template.scope}', '${templateId}')">
                                Generate
                            </button>
                        </td>
                    `;
                });
            } catch (error) {
                console.error('Error loading templates:', error);
            }
        }

        // Initial load
        loadTemplates();

        // Helper function for generating code
        async function generateCodeForTemplate(scope, templateId) {
            try {
                const generatedCode = await fetchData(`generate_api?scope=${scope}&template_id=${templateId}`, 'GET');
                document.getElementById('generatedCodeDisplay').textContent = generatedCode;
                document.getElementById('generatedCodeArea').classList.remove('hidden');
            } catch (error) {
                console.error('Error generating code:', error);
            }
        }
    </script>
</body>
</html>"""
