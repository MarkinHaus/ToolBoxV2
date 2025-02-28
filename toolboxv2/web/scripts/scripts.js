import {
    WebGLRenderer,
    Scene,
    PerspectiveCamera,
    MeshPhysicalMaterial,
    AmbientLight,
    PointLight,
    Group,
    TetrahedronGeometry,
    Mesh

} from "/web/node_modules/three/src/Three.js";

// Create a WebGLRenderer
export const renderer = new WebGLRenderer({ antialias: true });
let groops = []

let mashs = []

// Create a scene
const scene = new Scene();
// Create a camera
const camera = new PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.1, 1000);


// Create a milk glass material
const milkGlassMaterial = new MeshPhysicalMaterial({
    color: 0xffffff, // Set color to blue
    metalness: 0.5,
    roughness: 0.5,
    transparent: true,
    clearcoat: 0.5,
    clearcoatRoughness: 0.8,
    reflectivity: 0.4,
    opacity: 0.6
});

let triangleGeometry = createSierpinskiTriangle(5, 12);
let sk = 0.08;
let animantionFactorKlick = 8;
let animantionFactorIdeal = 12;
let animantionFactor = animantionFactorIdeal;
let sk2 = sk;

// Mouse interaction
let mouseX = 0;
let mouseY = 0;

let animationX = 0.002;
let animationY = 0.002;
let animationZ = 0.002;

let aspectr = 12;

let cam_zoom_val = 0;

let pointLightSto = []
let ambientLightSto = new AmbientLight(0x181823);

function uinistallSw(){
    navigator.serviceWorker.getRegistrations().then(function(registrations) {
        registrations.forEach(function(registration) {
            registration.unregister().then(function(success) {
                if (success) {
                    console.log('ServiceWorker mit dem Namen', registration.scope, 'wurde erfolgreich entfernt.');
                } else {
                    console.log('Fehler beim Entfernen des ServiceWorker mit dem Namen', registration.scope);
                }
            });
        });
    });
}


function initRenderer() {
    camera.position.z = 10;
    camera.position.y = 3.2;
    renderer.setSize(window.innerWidth, window.innerHeight);
    scene.add(triangleGeometry);
    loadDarkModeState()


// Create a point light
    const pointLight = new PointLight(0x181823, 1, 100);
    pointLight.position.set(2, -2, 2);
    pointLightSto.push(pointLight)
    scene.add(pointLight);

// Create a point light 0x537FE7 ligth
    const pointLight2 = new PointLight(0xffffff, 1, 100);
    pointLight2.position.set(2, 2, 2);
    pointLightSto.push(pointLight2)
    scene.add(pointLight2);

// Create an ambient light darg 0x181823
    const ambientLight = new AmbientLight(0x181823);
    scene.add(ambientLightSto);

}


initRenderer()

// Functions to load dark mode state
export function toggleDarkMode(init=false, theme='') {
    if (!init){
        theme = document.body.getAttribute("data-theme")
        theme = theme === "light" ? "dark" : "light"
        document.body.setAttribute('data-theme', theme)
        sessionStorage.setItem("darkModeStatus", theme);

    }else{
        document.body.setAttribute('data-theme', theme)
        sessionStorage.setItem("darkModeStatus", theme);
    }
    loadDarkModeState(theme)
}

function loadDarkModeState(theme) {
    let darkModeStatus = theme? theme: sessionStorage.getItem("darkModeStatus");
    let color = 0x181823
    if (darkModeStatus === "dark") {
        document.body.classList.add("dark-mode");
        renderer.setClearColor(0x000000)
        scene.remove(ambientLightSto)
        color = 0x181823

    } else {
        document.body.classList.remove("dark-mode");
        renderer.setClearColor(0xcccccc)
        color = 0x537FE7

    }
    scene.remove(ambientLightSto)
    ambientLightSto = new AmbientLight(color);
    scene.add(ambientLightSto)
    let position = [0,0,0]
    let new_pointLightSto = []
    for (let i = 0; i < pointLightSto.length; i++) {
        position = [pointLightSto[i].position.x,pointLightSto[i].position.y,pointLightSto[i].position.z];
        scene.remove(pointLightSto[i]);
        const pointLight = new PointLight(color, 1, 100);
        color = 0xffffff
        pointLight.position.set(position[0],position[1],position[2]);
        new_pointLightSto.push(pointLight);
        scene.add(pointLight);
    }
    pointLightSto = new_pointLightSto;
}

// Animation loop
function animateBD() {

    requestAnimationFrame(animateBD);
    triangleGeometry.rotateX(animationX/animantionFactor)
    triangleGeometry.rotateY(animationY/animantionFactor)
    triangleGeometry.rotateZ(animationZ/animantionFactor)

    for (let i = 0; i < groops.length; i++) {
        groops[i].rotateX(animationX/animantionFactor)
        groops[i].rotateY(animationY/animantionFactor)
        groops[i].rotateZ(animationZ/animantionFactor)
    }

    renderer.render(scene, camera);
}

/**
 * Create a Sierpinski Triangle using Three.js
 * @param {number} depth - The depth of recursion for the Sierpinski Triangle
 * @param {number} size - The size of the initial equilateral triangle
 * @returns {Group} - A group containing the Sierpinski Triangle geometry
 */
function createSierpinskiTriangle(depth, size) {
    const group = new Group();

    function createSierpinski(depth, size, group) {
        if (depth === 0) {
            const triangle = new TetrahedronGeometry(size*0.6);
            const mash = new Mesh(triangle, milkGlassMaterial)
            mashs.push(mash)
            group.add(mash);
        } else {
            const newSize = size / 2;
            const newDepth = depth - 1;
            const offset = newSize * Math.sqrt(3) / 2;

            const bottomLeftGroup = new Group();
            bottomLeftGroup.position.set(-offset/ 2, 0, -newSize / 4);
            createSierpinski(newDepth, newSize, bottomLeftGroup);
            group.add(bottomLeftGroup);

            const bottomRightGroup = new Group();
            bottomRightGroup.position.set(offset/ 2, 0, -newSize / 4);
            createSierpinski(newDepth, newSize, bottomRightGroup);
            group.add(bottomRightGroup);

            const bottomBackGroup = new Group();
            bottomBackGroup.position.set(0, 0, newSize / 2);
            createSierpinski(newDepth, newSize, bottomBackGroup);
            group.add(bottomBackGroup);

            const topGroup = new Group();
            topGroup.position.set(0, newSize * Math.sqrt(2 / 3), 0);
            createSierpinski(newDepth, newSize, topGroup);
            group.add(topGroup);

            if (depth !== 1){
                groops.push(topGroup)
                groops.push(bottomBackGroup)
                groops.push(bottomRightGroup)
                groops.push(bottomLeftGroup)
            }
        }
    }

    createSierpinski(depth, size, group);
    return group;
}


function endBgInteract() {
    animantionFactor = animantionFactorIdeal;
    animationX = 0.002;
    animationY = 0.002;
    animationZ = 0.002;
}


export function Set_animation_xyz(x,y,z,s=animantionFactorKlick+2){
    animantionFactor = s
    animationX = x
    animationY = y
    animationZ = z
}
export function Set_zoom(x){

    camera.position.z += x * (sk2/2);
    if (camera.position.z <= -2) {
        camera.position.z = 12
    }
    if (camera.position.z > 12) {
        camera.position.z = -2
    }
    camera.updateProjectionMatrix();
}


function updateSlider(slideAmount, axis)
{
    slideAmount = slideAmount/100
    if (axis === 0){
        console.log("Udateing X", slideAmount)
        for (let i = 0; i < mashs.length; i++) {
            mashs[i].rotateX(slideAmount)
        }
    }
    if (axis === 1){
        console.log("Udateing Y", slideAmount)
        for (let i = 0; i < mashs.length; i++) {
            mashs[i].rotateY(slideAmount)
        }
    }
    if (axis === 2){
        console.log("Udateing Z", slideAmount)
        for (let i = 0; i < mashs.length; i++) {
            mashs[i].rotateZ(slideAmount)
        }
    }
}


window.onwheel = e => {

    if(e.deltaY >= 0){
        // Scrolling Down with mouse
        camera.position.z -= sk2/2;
    } else {
        camera.position.z += sk2/2;
    }

    camera.updateProjectionMatrix();

    if (camera.position.z <= -2) {
        camera.position.z = 12
    }

    if (camera.position.z >= 12) {
        camera.position.z = -2
    }

    // if (20 > camera.position.z < 60 ){
    //     sk2 = sk*50;
    // } else if ( 12 > camera.position.z < 20){
    //     sk2 = sk*2;
    // } else if (-2 >camera.position.z < 12){
    //     sk2 = sk;
    // }

}


// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Call the function to load the dark mode state when entering a file
document.addEventListener('DOMContentLoaded', () => {

    const slideX = document.getElementById("slideX");
    if (slideX !== null){
        slideX.addEventListener("change", function (e)  {
            updateSlider(e.target.value, 0)
        });
    }
    const slideY = document.getElementById("slideY");
    if (slideY !== null){
        slideY.addEventListener("change", function (e)  {
            updateSlider(e.target.value, 1)
        });
    }
    const slideZ = document.getElementById("slideZ");
    if (slideZ !== null){
        slideZ.addEventListener("change", function (e)  {
            updateSlider(e.target.value, 2)
        });
    }
    const slideA = document.getElementById("slideA");
    if (slideA !== null){
        slideA.addEventListener("change", function (e)  {

            scene.remove(triangleGeometry)
            triangleGeometry = createSierpinskiTriangle(e.target.value, 12);
            scene.add(triangleGeometry)

        });
    }

});

export const addRenderer = () => {



const threeDScene = document.getElementById('threeDScene')
if (threeDScene){
threeDScene.appendChild(renderer.domElement);}
}
addRenderer()

setTimeout(()=>{
const s = document.getElementsByClassName('loaderCenter')
    if (s){if(s[0]){s[0].classList.add("none")}}}, 122)

let isMouseDown = false;
let startX = 0;
let startY = 0;
let currentX = 0;
let currentY = 0;

export function StartBgInteract(event) {
    isMouseDown = true;
    animantionFactor = animantionFactorKlick;

    // Capture the starting position
    if (event) {
        if (event.type.includes('mouse')) {
            startX = event.clientX;
            startY = event.clientY;
        } else if (event.type.includes('touch')) {
            startX = event.touches[0].clientX;
            startY = event.touches[0].clientY;
        }

        // Normalize coordinates (-1 to 1)
        const normalizedX = (startX / window.innerWidth) * 2 - 1;
        const normalizedY = -((startY / window.innerHeight) * 2 - 1);

        // Initial rotation based on click/touch position
        animationX = normalizedY * 0.02;
        animationY = normalizedX * 0.02;
        animationZ = (normalizedX + normalizedY) * 0.01;
    }
}

function handleBgMove(event) {
    if (!isMouseDown) return;

    // Get current position
    if (event.type.includes('mouse')) {
        currentX = event.clientX;
        currentY = event.clientY;
    } else if (event.type.includes('touch')) {
        currentX = event.touches[0].clientX;
        currentY = event.touches[0].clientY;
    }

    // Calculate movement delta from starting position
    const deltaX = currentX - startX;
    const deltaY = currentY - startY;

    // Convert delta to rotation values (scaled appropriately)
    // Vertical movement affects X rotation, horizontal affects Y rotation
    animationX = (deltaY / window.innerHeight) * Math.PI / 25;
    animationY = (deltaX / window.innerWidth) * Math.PI / 25;
    animationZ = ((deltaX + deltaY) / (window.innerWidth + window.innerHeight)) * Math.PI / 25;
}

export function EndBgInteract() {
    isMouseDown = false;
    animantionFactor = animantionFactorIdeal;

    // Reset to small default animation values
    animationX = 0.002;
    animationY = 0.002;
    animationZ = 0.002;
}

// Mouse event listeners
document.body.addEventListener('mousedown', (event) => {
    StartBgInteract(event);
});

document.body.addEventListener('mousemove', (event) => {
    handleBgMove(event);
});

document.body.addEventListener('mouseup', () => {
    EndBgInteract();
});

// Touch event listeners
document.body.addEventListener('touchstart', (event) => {
    StartBgInteract(event);
});

document.body.addEventListener('touchmove', (event) => {
    handleBgMove(event);
});

document.body.addEventListener('touchend', () => {
    EndBgInteract();
});

animateBD();


