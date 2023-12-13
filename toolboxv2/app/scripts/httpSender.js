
export const ToolBoxError = {
    none: "none",
    input_error: "InputError",
    internal_error: "InternalError",
    custom_error: "CustomError"
};

export const ToolBoxInterfaces = {
    cli: "CLI",
    api: "API",
    remote: "REMOTE",
    native: "NATIVE"
};

// Equivalent of Python dataclass in JavaScript
export class ToolBoxResult {
    constructor(data_to = ToolBoxInterfaces.cli, data_info = null, data = null) {
        this.data_to = data_to;
        this.data_info = data_info;
        this.data = data;
    }
}

export class ToolBoxInfo {
    constructor(exec_code, help_text) {
        this.exec_code = exec_code;
        this.help_text = help_text;
    }
}

export class Result {
    constructor(error = ToolBoxError.none, result = new ToolBoxResult(), info = new ToolBoxInfo(-1, "")) {
        this.error = error;
        this.result = result;
        this.info = info;
    }

    static default(TB_interface = ToolBoxInterfaces.cli) {
        return new Result(ToolBoxError.none, new ToolBoxResult(TB_interface), new ToolBoxInfo(-1, ""));
    }

    get() {
        return this.result.data;
    }
}

// Modified httpPostUrl function
export function httpPostUrl(module_name, function_name, parm, val, errorCallback, successCallback) {
    fetch('/api/' + module_name + '/' + function_name + '?' + parm + '=' + val, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    })
        .then(response => response.json())
        .then(data => {
            let result = new Result(data.error, new ToolBoxResult(data.result.data_to, data.result.data_info, data.result.data), new ToolBoxInfo(data.info.exec_code, data.info.help_text));
            if (result.error !== ToolBoxError.none) {
                // Handle error case
                return errorCallback(result);
            } else {
                // Handle success case
                return successCallback(result);
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            return errorCallback(new Result(ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, error.toString())));
        });
}
export function httpPostData(module_name, function_name, data, errorCallback, successCallback) {
    fetch('/api/' + module_name + '/' + function_name, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(data => {
            let result = new Result(data.error, new ToolBoxResult(data.result.data_to, data.result.data_info, data.result.data), new ToolBoxInfo(data.info.exec_code, data.info.help_text));
            if (result.error !== ToolBoxError.none) {
                // Handle error case
                return errorCallback(result);
            } else {
                // Handle success case
                return successCallback(result);
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            return errorCallback(new Result(ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, error.toString())));
        });
    return new Result(ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, error.toString()))
}

