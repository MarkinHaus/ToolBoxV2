function createOptionElement(option, index, total) {
    const optionElement = document.createElement('div');
    optionElement.classList.add('circle', `option${index}`);
    optionElement.style.backgroundColor = option.color;
    optionElement.style.transform = 'scale(0)';
    optionElement.addEventListener('click', (event) => {
        event.stopPropagation();
        console.log(option.return);
    });
    return optionElement;
}

function optionWheel(options) {
    const wheel = document.getElementById('option-wheel');
    const optionKeys = Object.keys(options);
    const totalOptions = optionKeys.length;

    optionKeys.forEach((key, index) => {
        const option = options[key];
        const optionElement = createOptionElement(option, index, totalOptions);
        wheel.appendChild(optionElement);
    });

    wheel.onclick = () => {
        wheel.classList.toggle('expanded');
    };
}




var segments = document.getElementsByClassName('segment');
var center = document.getElementById('center');

center.addEventListener('click', function() {
    for (var i = 0; i < segments.length; i++) {
        segments[i].style.opacity = 1;
    }
    this.style.display = 'none';
});

for (var i = 0; i < segments.length; i++) {
    segments[i].addEventListener('click', function() {
        for (var j = 0; j < segments.length; j++) {
            segments[j].classList.remove('selected');
        }
        this.classList.add('selected');
    });
}
