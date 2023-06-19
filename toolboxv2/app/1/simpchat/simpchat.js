

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById("chat-form").addEventListener("submit", async (event) => {
        event.preventDefault();
        const messageInput = document.getElementById("message-input");
        const messageText = messageInput.value.trim();

        if (messageText) {
            displayMessage(messageText, "user-message");
            displayLoadingSpinner(true);
            const response = await sendMessage(messageText);
            displayLoadingSpinner(false);
            displayMessage(response.res, "bot-message");
            messageInput.value = "";
        }
    });
});
async function sendMessage(text) {
    const response = await fetch("/api/post/isaa/run/run", {
        method: "POST",
        headers: {
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "token": "your_token_here",
            "data": {
                "name": "isaa-chat-web",
                "text": text
            }
        })
    });

    return await response.json();
}

function displayMessage(text, className) {
    const chatMessages = document.getElementById("chat-messages");
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", className);
    messageElement.textContent = text;
    chatMessages.prepend(messageElement);
}

function displayLoadingSpinner(show) {
    const loadingSpinner = document.getElementById("loading-spinner");
    loadingSpinner.style.display = show ? "block" : "none";
}
