

function doSomething1() {
    console.log('Do something1');
}
function doSomething() {
    console.log('Do something');
}
function addBalloon(element_id, flag, content, options) { // element_id flag:0 System 1 Isaa content: ['text','newline'] options: [['text', function]]
    console.log("ADDING Ballon ", element_id, flag)
    let speechBalloon= createSpeechBalloon(flag, content, options);

    const targetElement = document.getElementById(element_id);

    if (targetElement === null){
        console.log("Element :", element_id, "Not found")
        return
    }

    if (targetElement === undefined){
        console.log("Element :", element_id, "Not found")
        return
    }

    targetElement.appendChild(speechBalloon);

    function createSpeechBalloon(flag_set, content, options) {
        const template = document.getElementById('speech-balloon-template');
        const balloon = template.content.cloneNode(true).querySelector('.speech-balloon');
        const flag = flag_set ? "Isaa" : "System"

        const fromElement = balloon.querySelector('.speech-balloon-from');
        fromElement.textContent = flag;

        const closeButton = balloon.querySelector('.speech-balloon-close-button');
        closeButton.addEventListener('click', closeBalloon);

        const contentElement = balloon.querySelector('.speech-balloon-content');

        content.forEach(text => {
            const p_e = document.createElement('p');
            p_e.textContent = text;
            contentElement.appendChild(p_e);
        });

        const optionsElement = balloon.querySelector('.speech-balloon-options');
        options.forEach(option => {
            const button = document.createElement('button');
            button.textContent = option[0];
            button.addEventListener('click', ()=>{option[1]();closeBalloon(false)});
            optionsElement.appendChild(button);
        });

        const progressBar = balloon.querySelector('.speech-balloon-progress-bar');
        progressBar.addEventListener('animationend', closeBalloon);

        document.addEventListener('keydown', (event) => {
            options.forEach(option => {
                if (option[0].startsWith(event.key)) {
                    option[1]();
                    closeBalloon(false);
                }
            });
        });

        function closeBalloon(do_dot=true) {
            balloon.style.animation = 'speech-balloon-fadeOut 0.5s';
            progressBar.style.animationPlayState = 'paused';

            let remove = true

            setTimeout(() => {

                balloon.style.display = 'none'

                if (do_dot) {
                    const dot = document.createElement('div');
                    dot.classList.add('speech-balloon-dot');
                    dot.addEventListener('click', ()=>{
                        remove = false;
                        console.log("CLICK")
                        balloon.style.animation.replace('speech-balloon-fadeOut', 'speech-balloon-fadeIn');
                        balloon.style.display = 'flex'
                    })
                    if (flag==="System"){
                        dot.classList.add("speech-balloon-sys")
                    }else {
                        dot.classList.add("speech-balloon-isaa")
                    }
                    targetElement.appendChild(dot);

                    setTimeout(() => {
                        targetElement.removeChild(dot);
                        if (remove){
                            console.log("removed", remove)
                            targetElement.removeChild(balloon);
                            speechBalloon = null;
                        } else {
                            console.log("not removed", remove)
                        }
                    }, 15000);

                }else {
                    targetElement.removeChild(balloon);
                    speechBalloon = null;
                }


            }, 500);
        }

        if (flag==="System"){
            balloon.classList.add("speech-balloon-sys")
            progressBar.classList.add("speech-balloon-progress-sys")
        }else {
            balloon.classList.add("speech-balloon-isaa")
            progressBar.classList.add("speech-balloon-progress-isaa")
        }

        return balloon;
    }

}


