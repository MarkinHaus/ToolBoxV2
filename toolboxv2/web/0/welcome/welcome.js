import {getContent} from "../../scripts/WorkerSocketRenderer";

let currentStep = 0;

function WelcomeInit(){


    const greeting = document.getElementById('greeting');
    const explanation = document.getElementById('explanation');
    const greeting2 = document.getElementById('greeting2');
    const explanation2 = document.getElementById('explanation2');
    const options = document.getElementById('options');
    const option1 = document.getElementById('option1');
    const option2 = document.getElementById('option2');
    const next = document.getElementById('next');
    const inputContainer = document.getElementById('input-container');

    function switchText() {
        console.log(currentStep)

        if (currentStep === 0) {
            greeting.classList.add('fade');
            explanation.classList.add('fade2');

            setTimeout(() => {
                explanation2.classList.remove('none');
                greeting2.classList.remove('none');

                explanation2.classList.add('fade_in');
                greeting2.classList.add('fade_in');

                explanation.classList.add('none');
                greeting.classList.add('none');

                setTimeout(() => {
                    greeting.remove()
                    explanation.remove()
                }, 50);
            }, 2150);

        } else if (currentStep === 1) {
            greeting2.classList.add('fade');
            explanation2.classList.add('fade2');
            next.classList.add('fade');

            setTimeout(() => {
                explanation2.classList.add('none');
                greeting2.classList.add('none');
                next.classList.add('none');

                options.classList.add('fade_in');
                options.classList.remove('fade');
                options.style.display = 'block';

                setTimeout(() => {
                    greeting2.remove()
                    explanation2.remove()
                }, 50);
            }, 2150);


        }else if (currentStep === 2) {
            next.classList.remove('none');
            next.classList.remove('fade');
            next.classList.add('fade_in');
            setTimeout(() => {
                options.remove()
                option1.remove()
                option2.remove()
            }, 250);



        }else if (currentStep === 3) {
            next.classList.add('fade');
            inputContainer.classList.add('fade');
            setTimeout(() => {
                inputContainer.classList.add('none');
                next.classList.add('none');
                setTimeout(() => {
                    inputContainer.remove()
                    next.remove()
                }, 50);
            }, 250);


            localStorage.setItem('tasks', JSON.stringify([{ name: "Welcome", "task":{
                'use': "agent",
                    'name': "self",
                    'args': "Please introduce yourself to the user. Tell the user abut yur goals and capabilities. add 2 ideas for the user to start using you",
                    'return': "$ret"
            }}]));


        } else {
            currentStep--;
        }

        currentStep++;

    }

    function openStore(){
        switchText()
    }

    async function startWork(){
        switchText()
        console.log("startWork:getWidgetNave-getDrag")
        addBalloon("NAV", 0, ["Start Installation of first work Station", "installing Widgets"],[])
        getContent("getWidgetNave");
        addBalloon("WidgetNavCircle", 0, ["Widgets Navigation Installed", "next installation chat" ],[])
        addBalloon("SimpChatContainer", 0, ["chat Installed", "next and last installation Drag" ],[])
        getContent("getDrag");
        addBalloon("NAV", 0, ["chat Installed", "press on Init Draggable to start Dragging elements" ],[])
        document.getElementById('main').classList.remove('main-content')
    }


    function showInputContainer(do_install) {
        switchText()
        if (do_install){
            console.log("showInputContainer:installMod-isaa")
            WS.send(JSON.stringify({"ServerAction":"installMod","name":"isaa"}));
            next.removeEventListener('click', switchText, true);
            next.addEventListener('click',startWork, true);
            document.getElementById("EndP").innerText = "Please wait until the isaa setup process starts after completion press next"
        }else {
            addBalloon("input-container", 0, ["I see you decided not to use isaa or ended up here by accident.","that's not a problem isaa is an extension just like any other tool.", " You can add extensions at any time.", "You can also just look around in the store like so"], [["go to Store", openStore]])
            next.removeEventListener('click', switchText, true);
            next.addEventListener('click', openStore, true);
        }

        localStorage.setItem("local_ws.onopen:installMod-welcome", "false")
        options.classList.add('fade');

        setTimeout(() => {
            options.classList.add('none');
            inputContainer.classList.remove('fade');
            inputContainer.style.display = 'block';
        }, 1550)

        function test(){
            window.location.href = "www.google.com"
        }
    }

    option1.addEventListener('click', ()=> {
        showInputContainer(true)
    });
    option2.addEventListener('click', ()=> {
        showInputContainer(false)
    });
    next.addEventListener('click',switchText, true);

}

async function getsMSG(){
    console.log("startWork:getsMSG")
    getContent("getsMSG")
}

getsMSG().then(r => {
    WelcomeInit()
})
