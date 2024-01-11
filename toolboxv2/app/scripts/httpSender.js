
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


    get() {
        return this.result.data;
    }
}

function wrapInResult(data) {
    let result = new Result("httpPostData Error No date Returned", new ToolBoxResult(ToolBoxInterfaces.native, "Es wurden keine darten vom server zurück gegeben", {}), new ToolBoxInfo(-986, "NO data"));

    function valid_val(v) {
        return  v !== undefined && v !== null
    }

    let data_to = ""
    let data_info = ""
    let r_data;
    let exec_code = -989
    let help_text = ""

    if (valid_val(data)){
        const error = data.error
        if (valid_val(data.result)){
            data_to = data.result.data_to;
            data_info = data.result.data_info;
            r_data = data.result.data;
        }
        if (valid_val(data.info)){
            exec_code = data.info.exec_code;
            help_text = data.info.help_text;
        }
        result = new Result(error, new ToolBoxResult(data_to, data_info, r_data), new ToolBoxInfo(exec_code, help_text));

    }
    return result
}

// Modified httpPostUrl function
export function httpPostUrl(module_name, function_name, params, errorCallback, successCallback) {
    fetch('/api/' + module_name + '/' + function_name + '?' + params, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    })
        .then(response => response.json())
        .then(data => {
            const result = wrapInResult(data)
            if (result.error !== ToolBoxError.none) {
                // Handle error case
                return errorCallback(result);
            } else {
                // Handle success case
                return successCallback(result);
            }
        })
        .catch((error) => {
            return errorCallback(new Result(ToolBoxError.internal_error, new ToolBoxResult("Frontend Dev", "Error in successCallback at : "+module_name+'.'+ function_name, error), new ToolBoxInfo(-1, error.toString())));
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

            const result = wrapInResult(data)
            if (result.error !== ToolBoxError.none) {
                // Handle error case
                return errorCallback(result);
            } else {
                // Handle success case
                return successCallback(result);
            }
        })
        .catch((error) => {
            return errorCallback(new Result(ToolBoxError.internal_error, new ToolBoxResult("Frontend Dev", "Error in successCallback at : " +module_name+'.'+ function_name, error), new ToolBoxInfo(-1, error.toString())));
        });
    // return new Result(ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, 'intern error client side 90975'))
}

