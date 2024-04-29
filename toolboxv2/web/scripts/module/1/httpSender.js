import {rendererPipeline} from "./WorkerSocketRenderer.js";

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
    constructor(origin = [], error = ToolBoxError.none, result = new ToolBoxResult(), info = new ToolBoxInfo(-1, "")) {
        this.origin = origin;
        this.error = error;
        this.result = result;
        this.info = info;
    }

    log(){
        console.log(
            "======== Result ========\nFunction Exec coed:",
            this.info.exec_code,
            "\nInfo's:",this.info.help_text, "<|>", this.result.data_info,
            "\nOrigin:",this.origin,
            "\nData_to:",this.result.data_to,
            "\nData:",this.result.data,
            "\nerror:",this.error,
            "\n------- EndOfD -------",
        )
    }

    get() {
        return this.result.data;
    }
}

function wrapInResult(data, from_string=false) {
    let result = new Result(["local", "!", "no-data-parsed"], "httpPostData Error No date Returned", new ToolBoxResult(ToolBoxInterfaces.native, "Es wurden keine darten vom server zurück gegeben", {}), new ToolBoxInfo(-986, "NO data"));

    function valid_val(v) {
        //console.log("V::",v)
        return  v !== undefined && v !== null
    }

    let data_to = ""
    let data_info = ""
    let r_data;
    let exec_code = -989
    let help_text = ""

    if (!from_string) {
        if (valid_val(data)) {
            const error = data.error
            const origen = data.origin
            console.log("[origin]:", origen)
            if (valid_val(data.result)) {
                data_to = data.result.data_to;
                data_info = data.result.data_info;
                r_data = data.result.data;
            }
            if (valid_val(data.info)) {
                exec_code = data.info.exec_code;
                help_text = data.info.help_text;
            }
            result = new Result(origen, error, new ToolBoxResult(data_to, data_info, r_data), new ToolBoxInfo(exec_code, help_text));

        }
    } else {
        try {
            data = JSON.parse(data)
            const error_ = data["error"]
            const origen = data["origin"]
            console.log("[s-origin]:", origen)
            if (valid_val(data["result"])) {
                data_to = data["result"]["data_to"];
                data_info = data["result"]["data_info"];
                r_data = data["result"]["data"];
            }
            if (valid_val(data["info"])) {
                exec_code = data["info"]["exec_code"];
                help_text = data["info"]["help_text"];
            }
            result = new Result(origen, error_, new ToolBoxResult(data_to, data_info, r_data), new ToolBoxInfo(exec_code, help_text));
        }catch (e){
            console.log("Error Parsing Result")
        }
    }

    return result
}

// Modified httpPostUrl function
export function httpPostUrl(module_name, function_name, params, errorCallback, successCallback, from_string=false) {
    fetch('/api/' + module_name + '/' + function_name + '?' + params, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    })
        .then(response => response.json())
        .then(data => {
            console.log("DATA:", data)
            const result = wrapInResult(data, from_string)
            result.log()
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

export function AuthHttpPostData(username, errorCallback, successCallback) {
    fetch('/validateSession', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            'Jwt_claim': window.localStorage.getItem('jwt_claim_device'),
            'Username':username})
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


async function handleHtmxAfterRequest(event) {
    let xhr = event.detail.xhr; // Der XMLHttpRequest
    let response = xhr.response;

    let json = {}

    try {
        // Versuchen Sie, die Antwort als JSON zu parsen
        json = JSON.parse(response);

    } catch (e) {
        // Wenn ein Fehler auftritt, handelt es sich wahrscheinlich nicht um eine JSON-Antwort
        // oder das JSON-Format entspricht nicht der Erwartung
        console.log("Invalid JSON error", e)
        return "Error"
    }

    try {
        // Versuchen Sie, die Antwort als JSON zu parsen
        const result = wrapInResult(json, true)
        result.log()

        console.log("result:", result.origin.at(2) === 'REMOTE')
        console.log(result.get().toString().startsWith('<'), result.origin.at(2))

        if (result.error !== ToolBoxError.none) {
            // Handle error case
            return "errorCallback(result);"
        } else if (result.get().toString().startsWith('<')) {
            // Handle success case
            if (event.detail && event.detail.target) {
                console.log("event.detail.target", event.detail.target)
                var target = event.detail.target
                target.innerHTML = result.get();
            }
            return "successCallback(result);"
        }else if (result.origin.at(2) === 'REMOTE') {
            await rendererPipeline(result.get()).then(r => {
                console.log("Rendering Don")
                return "successCallback(result);"
            })
        }

    } catch (e) {
        console.log("Result Parsing error", e)
    }
}
let isHtmxAfterRequestListenerAdded = false;
export function resultHtmxWrapper(){
    if (!isHtmxAfterRequestListenerAdded){
        try{
            document.body.removeEventListener('htmx:afterRequest', handleHtmxAfterRequest);
            isHtmxAfterRequestListenerAdded = false
        }
        catch (e){
            console.log("Fist init handleHtmxAfterRequest")
        }
    }

    if (!isHtmxAfterRequestListenerAdded) {
        console.log("ADD HTMX LISSENER TO body")
        document.body.addEventListener('htmx:afterRequest', handleHtmxAfterRequest);
        isHtmxAfterRequestListenerAdded = true;
    }


}

export function getContent(path, name='DashProvider', parms=''){
    try{
        if (WS !== undefined){
            WS.send(JSON.stringify({"ServerAction":path}))
        }
    }catch (e){
        httpPostUrl(name, path, parms, (result)=>{
            result.log()
        }, (result)=>{
            rendererPipeline(result.get()).then(r => {
                console.log("[Don redering]")
            })
        }, true)
    }

}
