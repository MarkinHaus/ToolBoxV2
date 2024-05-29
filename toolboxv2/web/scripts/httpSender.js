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

    html(){
        return `<div style="background-color: var(--background-color);
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    padding: 2px;">
  <div style="background-color: var(--background-color);
    padding: 5px;text-align:center">
    <p>======== Result ========</p>
    <p>Function Exec code: <span id="execCode">`+this.info.exec_code+`</span></p>
    <p>Info's: <span id="infoText">`+this.info.help_text+`</span> <|> <span id="dataInfo">`+this.result.data_info+`</span></p>
    <p>Origin: <span id="originText">`+this.origin+`</span></p>
    <p>Data_to: <span id="dataTo">`+this.result.data_to+`</span></p>
    <p>Data: <span id="data">`+this.result.data+`</span></p>
    <p>Error: <span id="errorText">`+this.error+`</span></p>
    <p>------- EndOfD -------</p>
  </div>
</div>`
    }

    get() {
        return this.result.data;
    }
}

export function wrapInResult(data, from_string=false) {
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
            if (typeof data === "string") {
                data = JSON.parse(data)
            }
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
            console.log(e)
            result = new Result(["local", "!", "data-parsed"], ToolBoxError.none, new ToolBoxResult("frontend", "ato-wrapped", data), new ToolBoxInfo(0, "generated in httpSender"));
        }
    }

    return result
}

async function parseResponse(response){
    // console.log(response)
    const res = await response.text()
    if (res.toString().startsWith("{")&&res.toString().endsWith("}")){
        return JSON.parse(res.toString())
    }
    return res
}

// Modified httpPostUrl function
export function httpPostUrl(module_name, function_name, params, errorCallback, successCallback, from_string=false) {
    // console.log("httpPostUrl:", module_name, function_name, params)
    return fetch('/api/' + module_name + '/' + function_name + '?' + params, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    })
        .then(async (data )=> {
                data = await parseResponse(data)
            // console.log("httpPostUrl: ",data, typeof data)
                if (from_string){data = data.toString()}
                const result = wrapInResult(data, typeof data === "string" || from_string)
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
    // console.log("httpPostData:", module_name, function_name)
    return fetch('/api/' + module_name + '/' + function_name, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(data => {
            setTimeout(async ()=>{
                data = await parseResponse(data)
                // console.log("httpPostData: ",data, typeof data)
                const result = wrapInResult(data, typeof data === "string")
                result.log()
                if (result.error !== ToolBoxError.none) {
                    // Handle error case
                    return errorCallback(result);
                } else {
                    // Handle success case
                    return successCallback(result);
                }
            })
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
            return errorCallback(new Result(ToolBoxError.internal_error, new ToolBoxResult("Frontend Dev", "Error in successCallback in AuthHttpPostData : " +username, error), new ToolBoxInfo(-1, error.toString())));
        });
    // return new Result(ToolBoxError.internal_error, new ToolBoxResult(), new ToolBoxInfo(-1, 'intern error client side 90975'))
}


