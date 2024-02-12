var currentStepIsaaInit = 0;

function IIInit(){
    const openaiHadder = document.getElementById('openaiHadder');
    const openaiExplanation = document.getElementById('openaiExplanation');
    const huggingfaceHadder = document.getElementById('huggingfaceHadder');
    const huggingfaceExplanation = document.getElementById('huggingfaceExplanation');
    const openaiInput = document.getElementById('openai-input');
    const huggingfaceInput = document.getElementById('huggingface-input');
    const next = document.getElementById('ButtonIsaaSetUPnext');
    const overlay = document.getElementById('ISaaSetUP');

    openaiInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            switchText()
            console.log('Enter-Taste wurde gedrückt');
        }
    });
    huggingfaceInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            switchText()
            console.log('Enter-Taste wurde gedrückt');
        }
    });

    function switchText() {
        console.log(currentStepIsaaInit)
        if (currentStepIsaaInit === 0) {
            openaiHadder.classList.add('fade');
            openaiExplanation.classList.add('fade2');
            openaiInput.classList.add('fade2');

            setTimeout(() => {
                huggingfaceExplanation.classList.remove('none');
                huggingfaceHadder.classList.remove('none');

                huggingfaceExplanation.classList.add('fade_in');
                huggingfaceHadder.classList.add('fade_in');

                openaiExplanation.classList.add('none');
                openaiHadder.classList.add('none');
                openaiInput.classList.add('none');

                setTimeout(() => {
                    openaiHadder.remove()
                    openaiExplanation.remove()
                    openaiInput.remove()
                }, 50);
            }, 2150);

        } else if (currentStepIsaaInit === 1) {
            huggingfaceHadder.classList.add('fade');
            huggingfaceExplanation.classList.add('fade2');

            setTimeout(() => {
                huggingfaceExplanation.classList.add('none');
                huggingfaceHadder.classList.add('none');
                huggingfaceInput.classList.add('fade_in');
                huggingfaceInput.classList.remove('fade');

                setTimeout(() => {
                    huggingfaceHadder.remove()
                    huggingfaceExplanation.remove()
                    overlay.remove()
                    setTimeout(() => {
                        sedData()
                    }, 500);
                }, 50);
            }, 2150);

        } else {
            currentStepIsaaInit--;
        }
        currentStepIsaaInit++;
    }

    function sedData(){
        switchText()
        let huggingface = huggingfaceInput.value
        let openai = openaiInput.value
        console.log("sedData:addConfig")
        WS.send(JSON.stringify({"ServerAction":"addConfig", "name":"isaa", "key":"OPENAI_API_KEY", "value":openai}));
        if (huggingface) {
            WS.send(JSON.stringify({"ServerAction":"addConfig", "name":"isaa", "key":"HUGGINGFACEHUB_API_TOKEN", "value":huggingface}));
        }
    }

    next.addEventListener('click', switchText, true);

}


IIInit()
