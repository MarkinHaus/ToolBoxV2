import {httpPostUrl} from "/web/scripts/httpSender.js";
import {Set_animation_xyz, EndBgInteract} from "/web/scripts/scripts.js";

document.addEventListener('DOMContentLoaded', () => {

    document.getElementById('email-subscription-form').addEventListener('submit', function(event) {
        event.preventDefault();

        const email = document.getElementById('email').value;
        Set_animation_xyz(0, 0.6, 0, 16)
        httpPostUrl("email_waiting_list", "add", "email=" + email,
            (result) => {
                console.log("[Error]",result)
                document.getElementById('infoText').textContent = result.info.help_text;
                document.getElementById('infoPopup').style.display = 'block';
                result.info.exec_code = -2;
                setTimeout(()=>{
                    Set_animation_xyz(1, 0.02, 0, 6)
                }, 300)
                setTimeout(()=>{
                    EndBgInteract()
                }, 300)
                return result;
            },
            (result) => {
                console.log("[Succses]",result)
                document.getElementById('infoText').textContent = result.info.help_text;
                document.getElementById('infoPopup').style.display = 'block';
                document.getElementById('success-message').hidden = false;
                result.info.exec_code = 1;
                setTimeout(()=>{
                    Set_animation_xyz(1, 0.02, 0, 6)
                }, 300)
                setTimeout(()=>{
                    EndBgInteract()
                }, 300)
                return result;
            }
        );
    });
});
